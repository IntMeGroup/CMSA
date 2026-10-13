import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import swanlab
from dataset import DatasetBuilder
from model import FR_MeshQA, NR_MeshQA
from utils import RankLoss, count_model_params, get_srcc_plcc


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                value.copy_(model_value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> nn.Module:
    pretrained = not args.no_pretrained
    if args.modeltype == "FR":
        return FR_MeshQA(embed_dim=args.embed_dim, pretrained=pretrained)
    if args.modeltype == "NR":
        return NR_MeshQA(embed_dim=args.embed_dim, pretrained=pretrained)
    raise ValueError(f"Unknown modeltype: {args.modeltype}")


def build_optimizer(model: nn.Module, lr: float, backbone_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if "visual" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr, "name": "head"})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def set_backbone_trainable(model: nn.Module, trainable_keywords: List[str]) -> None:
    for name, param in model.named_parameters():
        if "visual" not in name:
            param.requires_grad = True
            continue
        param.requires_grad = any(keyword in name for keyword in trainable_keywords)


def configure_training_stage(model: nn.Module, stage: str) -> None:
    if stage == "head":
        set_backbone_trainable(model, ["visual.norm"])
        return
    if stage == "late":
        set_backbone_trainable(model, ["visual.layers.3", "visual.norm"])
        return
    if stage == "full":
        set_backbone_trainable(model, ["visual.layers.0", "visual.layers.1", "visual.layers.2", "visual.layers.3", "visual.norm"])
        return
    raise ValueError(f"Unsupported training stage: {stage}")


def apply_stage_learning_rates(
    optimizer: torch.optim.Optimizer,
    stage: str,
    lr: float,
    backbone_lr: float,
) -> None:
    stage_backbone_lr = backbone_lr
    if stage == "head":
        stage_backbone_lr = backbone_lr * 0.5
    elif stage == "full":
        stage_backbone_lr = backbone_lr * 0.5

    for group in optimizer.param_groups:
        group_name = group.get("name", "head")
        group["lr"] = stage_backbone_lr if group_name == "backbone" else lr


def sync_scheduler_base_lrs(scheduler: torch.optim.lr_scheduler._LRScheduler, optimizer: torch.optim.Optimizer) -> None:
    scheduler.base_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def forward_model(model: nn.Module, batch: Dict[str, torch.Tensor], modeltype: str) -> torch.Tensor:
    if modeltype == "FR":
        pred = model(
            batch["dist_mosaic"],
            batch["dist_depth"],
            batch["dist_normal"],
            batch["dist_geometry"],
            batch["ref_mosaic"],
            batch["ref_depth"],
            batch["ref_normal"],
            batch["ref_geometry"],
        )
    else:
        pred = model(
            batch["dist_mosaic"],
            batch["dist_depth"],
            batch["dist_normal"],
            batch["dist_geometry"],
        )
    if pred.dim() == 0:
        pred = pred.unsqueeze(0)
    return pred


def move_batch_to_device(batch: Dict[str, Union[torch.Tensor, List[str]]], device: torch.device) -> Dict[str, Union[torch.Tensor, List[str]]]:
    output: Dict[str, Union[torch.Tensor, List[str]]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def outputs_to_csv(outputs: Dict[str, Union[List[str], torch.Tensor]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(
        {
            "mesh_path": outputs["names"],
            "pred_score": outputs["preds"].detach().cpu().numpy().reshape(-1),
            "gt_mos": outputs["targets"].detach().cpu().numpy().reshape(-1),
        }
    )
    df.to_csv(csv_path, index=False)


def run_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    reg_criterion: nn.Module,
    rank_criterion: nn.Module,
    device: torch.device,
    modeltype: str,
    rank_wgt: float,
    grad_accum_steps: int = 1,
    grad_clip: float = 0.0,
    optimizer: torch.optim.Optimizer = None,
    ema_model: ModelEma | None = None,
) -> Dict[str, Union[List[str], torch.Tensor, float]]:
    is_train = optimizer is not None
    num_samples = len(dataloader.dataset)
    epoch_outputs = {
        "names": [],
        "preds": torch.empty(0, device=device),
        "targets": torch.empty(0, device=device),
        "reg_loss": torch.empty(0, device=device),
        "rank_loss": torch.empty(0, device=device),
        "loss": torch.empty(0, device=device),
    }

    model.train(mode=is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        for step_idx, sample_batched in enumerate(dataloader):
            batch = move_batch_to_device(sample_batched, device)
            pred_quality = forward_model(model, batch, modeltype)
            targets = batch["score"]

            reg_loss = reg_criterion(pred_quality, targets)
            rank_loss = rank_criterion(pred_quality, targets)
            # loss = reg_loss + rank_wgt * rank_loss
            bias_loss = (pred_quality.mean() - targets.mean()).abs()
            loss = reg_loss + rank_wgt * rank_loss + 0.1 * bias_loss

            if is_train:
                (loss / grad_accum_steps).backward()
                should_step = ((step_idx + 1) % grad_accum_steps == 0) or (step_idx + 1 == len(dataloader))
                if should_step:
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    if ema_model is not None:
                        ema_model.update(model)
                    optimizer.zero_grad(set_to_none=True)

            batch_size = len(batch["name"])
            epoch_outputs["names"] += batch["name"]
            epoch_outputs["preds"] = torch.cat((epoch_outputs["preds"], pred_quality.detach()), dim=0)
            epoch_outputs["targets"] = torch.cat((epoch_outputs["targets"], targets.detach()), dim=0)
            epoch_outputs["reg_loss"] = torch.cat((epoch_outputs["reg_loss"], reg_loss.detach().unsqueeze(0) * batch_size), dim=0)
            epoch_outputs["rank_loss"] = torch.cat((epoch_outputs["rank_loss"], rank_loss.detach().unsqueeze(0) * batch_size), dim=0)
            epoch_outputs["loss"] = torch.cat((epoch_outputs["loss"], loss.detach().unsqueeze(0) * batch_size), dim=0)

    epoch_outputs["avg_reg_loss"] = epoch_outputs["reg_loss"].sum() / max(1, num_samples)
    epoch_outputs["avg_rank_loss"] = epoch_outputs["rank_loss"].sum() / max(1, num_samples)
    epoch_outputs["avg_loss"] = epoch_outputs["loss"].sum() / max(1, num_samples)
    srcc, plcc = get_srcc_plcc(preds=epoch_outputs["preds"], targets=epoch_outputs["targets"])
    epoch_outputs["srcc"] = float(srcc)
    epoch_outputs["plcc"] = float(plcc)
    epoch_outputs["rmse"] = compute_rmse(epoch_outputs["preds"], epoch_outputs["targets"])
    return epoch_outputs


def log_epoch(split: str, epoch: int, num_epochs: int, outputs: Dict[str, Union[List[str], torch.Tensor, float]], split_seed: int) -> None:
    logging.info(
        "%s Epoch [%d/%d] | reg: %.4f | rank: %.4f | loss: %.4f | RMSE: %.4f | SRCC: %.4f | PLCC: %.4f",
        split,
        epoch + 1,
        num_epochs,
        outputs["avg_reg_loss"].item(),
        outputs["avg_rank_loss"].item(),
        outputs["avg_loss"].item(),
        float(outputs["rmse"]),
        float(outputs["srcc"]),
        float(outputs["plcc"]),
    )
    if split == "Train":
        swanlab.log(
            {
                f"{split_seed}_loss": outputs["avg_loss"].item(),
                f"{split_seed}_rmse": float(outputs["rmse"]),
                f"{split_seed}_srcc": float(outputs["srcc"]),
                f"{split_seed}_plcc": float(outputs["plcc"]),
            }
        )
    else:
        swanlab.log(
            {
                f"{split_seed}_valloss": outputs["avg_loss"].item(),
                f"{split_seed}_valrmse": float(outputs["rmse"]),
                f"{split_seed}_valsrcc": float(outputs["srcc"]),
                f"{split_seed}_valplcc": float(outputs["plcc"]),
            }
        )


def main(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    trainset = DatasetBuilder().build(
        name=args.dataset,
        root_dir=args.root_dir,
        split="train",
        normalize_score=args.norm,
        seed=args.split_seed,
        kfold_seed=args.kfold_seed,
        random_patch_shuffle=not args.disable_patch_shuffle,
        patch_shuffle_prob=args.patch_shuffle_prob,
        view_dropout_prob=args.view_dropout_prob,
    )
    validset = DatasetBuilder().build(
        name=args.dataset,
        root_dir=args.root_dir,
        split="test",
        normalize_score=args.norm,
        seed=args.split_seed,
        kfold_seed=args.kfold_seed,
        random_patch_shuffle=False,
    )

    validloader = DataLoader(
        validset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    model = build_model(args).to(device)
    logging.info(model)

    reg_criterion = nn.MSELoss()
    rank_criterion = RankLoss()

    start_epoch = 0
    best_metrics = {"score": float("-inf"), "srcc": float("-inf"), "plcc": float("-inf"), "rmse": float("inf"), "best_epoch": -1}
    head_epochs = max(0, args.head_only_epochs)
    late_epochs = max(head_epochs, args.late_tune_epochs)
    early_stop_counter = 0

    if start_epoch < head_epochs:
        current_stage = "head"
    elif start_epoch < late_epochs:
        current_stage = "late"
    else:
        current_stage = "full"
    configure_training_stage(model, current_stage)
    logging.info("Initial training stage: %s", current_stage)
    logging.info(count_model_params(model))

    optimizer = build_optimizer(model, args.lr, args.backbone_lr, args.weight_decay)
    ema_model = ModelEma(model, args.ema_decay) if args.use_ema else None
    apply_stage_learning_rates(optimizer, current_stage, args.lr, args.backbone_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)
    sync_scheduler_base_lrs(scheduler, optimizer)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    ckpt_dir = os.path.join(args.save, "ckpt")
    pred_dir = os.path.join(args.save, "predictions")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs):
        next_stage = current_stage
        if epoch < head_epochs:
            next_stage = "head"
        elif epoch < late_epochs:
            next_stage = "late"
        else:
            next_stage = "late" if args.enable_late_tuning else "full"
        if next_stage != current_stage:
            current_stage = next_stage
            configure_training_stage(model, current_stage)
            apply_stage_learning_rates(optimizer, current_stage, args.lr, args.backbone_lr)
            sync_scheduler_base_lrs(scheduler, optimizer)
            logging.info("Switched training stage to %s at epoch %d", current_stage, epoch + 1)
            logging.info(count_model_params(model))

        trainset.set_epoch_k()
        logging.info("Trainset set k %s at epoch %d", trainset.k, epoch + 1)
        trainloader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        train_outputs = run_one_epoch(
            dataloader=trainloader,
            model=model,
            reg_criterion=reg_criterion,
            rank_criterion=rank_criterion,
            device=device,
            modeltype=args.modeltype,
            rank_wgt=args.rank_wgt,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            optimizer=optimizer,
            ema_model=ema_model,
        )
        eval_model = ema_model.module if ema_model is not None else model
        valid_outputs = run_one_epoch(
            dataloader=validloader,
            model=eval_model,
            reg_criterion=reg_criterion,
            rank_criterion=rank_criterion,
            device=device,
            modeltype=args.modeltype,
            rank_wgt=args.rank_wgt,
            optimizer=None,
        )

        scheduler.step()
        log_epoch("Train", epoch, args.num_epochs, train_outputs, args.split_seed)
        log_epoch("Valid", epoch, args.num_epochs, valid_outputs, args.split_seed)
        lrs = {group.get("name", f"group_{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
        logging.info("Epoch [%d/%d] | stage: %s | lr: %s | ema: %s", epoch + 1, args.num_epochs, current_stage, lrs, args.use_ema)

        current_score = float(valid_outputs["srcc"]) + float(valid_outputs["plcc"])
        if current_score > best_metrics["score"]:
            best_metrics = {
                "score": current_score,
                "srcc": float(valid_outputs["srcc"]),
                "plcc": float(valid_outputs["plcc"]),
                "rmse": float(valid_outputs["rmse"]),
                "best_epoch": epoch,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema_model.module.state_dict() if ema_model is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metrics": best_metrics,
                    "best_epoch": epoch,
                },
                os.path.join(ckpt_dir, f"{args.split_seed}_best.pth"),
            )

            outputs_to_csv(train_outputs, os.path.join(pred_dir, f"fold_{args.split_seed}_best_train.csv"))
            outputs_to_csv(valid_outputs, os.path.join(pred_dir, f"fold_{args.split_seed}_best_val.csv"))

            logging.info(
                "Updated best checkpoint | RMSE: %.4f | SRCC: %.4f | PLCC: %.4f",
                best_metrics["rmse"],
                best_metrics["srcc"],
                best_metrics["plcc"],
            )
            swanlab.log(
                {
                    f"{args.split_seed}_bestrmse": best_metrics["rmse"],
                    f"{args.split_seed}_bestsrcc": best_metrics["srcc"],
                    f"{args.split_seed}_bestplcc": best_metrics["plcc"],
                }
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            logging.info(
                "Early stopping triggered after %d epochs without validation improvement. Best epoch: %d",
                early_stop_counter,
                best_metrics["best_epoch"] + 1,
            )
            return best_metrics

    return best_metrics


def analyze_metrics(metrics_list, mode="avg"):
    if not metrics_list:
        return {}

    df = pd.DataFrame(metrics_list)

    if mode == "avg":
        return df.mean(numeric_only=True).to_dict()
    elif mode == "median":
        return df.median(numeric_only=True).to_dict()
    elif mode == "best":
        df["_temp_sum"] = df["plcc"] + df["srcc"]
        best_idx = df["_temp_sum"].idxmax()
        return df.loc[best_idx].drop("_temp_sum").to_dict()
    else:
        raise ValueError("mode 参数必须是 'avg'、'median' 或 'best'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal MeshQA 5-fold training")
    parser.add_argument("--dataset", type=str, default="TMQA")
    parser.add_argument("--root_dir", type=str, default="/DATA/DATA3/fukang/MQA/SJTU-TMQA-geotexdn")
    parser.add_argument("--save", type=str, default="exp")
    parser.add_argument("--modeltype", type=str, default="FR", choices=["FR", "NR"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_step", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-6)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=1)
    parser.add_argument("--kfold_seed", type=int, default=None)
    parser.add_argument("--rank_wgt", type=float, default=1.0)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--disable_patch_shuffle", action="store_true")
    parser.add_argument("--patch_shuffle_prob", type=float, default=0.5)
    parser.add_argument("--view_dropout_prob", type=float, default=0.1)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--head_only_epochs", type=int, default=20)
    parser.add_argument("--late_tune_epochs", type=int, default=30)
    parser.add_argument("--enable_late_tuning", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    args = parser.parse_args()

    args.save = os.path.join(args.save, time.strftime("%Y%m%d_%H%M%S") + f"_{args.dataset}_{args.modeltype}")
    os.makedirs(args.save, exist_ok=True)

    swanlab.init(project="m3qa_5folds_" + args.dataset, config=vars(args))

    split_seed_list = range(5)
    metrics_all = []
    log_format = "%(asctime)s %(message)s"

    for split_seed in split_seed_list:
        args.split_seed = split_seed
        print(args)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p", force=True)
        fh = logging.FileHandler(os.path.join(args.save, f"log_{args.split_seed}.txt"))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        for arg in vars(args):
            if arg != "root_dir":
                logging.info("%s: %s", arg, getattr(args, arg))

        best_metrics = main(args)
        best_metrics["fold"] = split_seed
        metrics_all.append(best_metrics)

        logging.getLogger().removeHandler(fh)
        fh.close()

    fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    for arg in vars(args):
        if arg != "root_dir":
            logging.info("%s: %s", arg, getattr(args, arg))
    logging.info(metrics_all)

    pd.DataFrame(metrics_all).to_csv(os.path.join(args.save, "five_fold_metrics.csv"), index=False)

    avg_res = analyze_metrics(metrics_all, mode="avg")
    print(f"📊 平均值结果: {avg_res}")
    logging.info(json.dumps(avg_res, ensure_ascii=False, indent=4, sort_keys=True))
    swanlab.log({"avg_res": swanlab.Text(str(avg_res))})

    best_res = analyze_metrics(metrics_all, mode="best")
    print(f"🏆 最佳轮次结果: {best_res}")
    logging.info(json.dumps(best_res, ensure_ascii=False, indent=4, sort_keys=True))
    swanlab.log({"best_res": swanlab.Text(str(best_res))})

    median_res = analyze_metrics(metrics_all, mode="median")
    print(f"🏆 中位数结果: {median_res}")
    logging.info(json.dumps(median_res, ensure_ascii=False, indent=4, sort_keys=True))
    swanlab.log({"median_res": swanlab.Text(str(median_res))})
