import csv
import gc
import json
import math
import warnings
from dataclasses import asdict, is_dataclass, replace as dc_replace
from pathlib import Path
from typing import Dict, Optional, Tuple
import random
import torch
import yaml
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..config import SimplifiedConfig
from ..core.logger import log_info
from ..data.dataset import create_dataloaders
from .loss import DetectionLoss
from .postprocess import nms, decode_bboxes
from .metrics import mean_avg_precision_metric
from .utils import EMAModel, EarlyStopping
from .export import export_onnx as _export_onnx


def relativize_config_paths(config: SimplifiedConfig) -> SimplifiedConfig:
    # Make model/pretrained paths relative to cwd before saving so checkpoints
    # stay portable across machines.
    cwd = Path.cwd()
    changes = {}
    for field_name in ("model", "pretrained"):
        value = getattr(config, field_name, None)
        if not value:
            continue
        p = Path(value)
        if p.is_absolute():
            try:
                changes[field_name] = str(p.relative_to(cwd))
            except ValueError:
                # outside cwd, keep as-is
                pass  
    return dc_replace(config, **changes) if changes else config


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 config: SimplifiedConfig) -> None:
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        self.loss_fn = DetectionLoss(
            model=model,
            img_size=(config.imgsz, config.imgsz),
            number_of_classes=getattr(model, "nc", config.extra.get("nc", 80)),
            reg_max=config.extra.get("reg_max", 16),
            strides=torch.tensor(config.extra.get("strides", [8, 16, 32]), device=self.device),
            box_gain=config.extra.get("box_gain", 0.05),
            cls_gain=config.extra.get("cls_gain", 0.5),
            dfl_gain=config.extra.get("dfl_gain", 1.0),
            topk=config.extra.get("topk", 10),
            expand_small_boxes=config.extra.get("expand_small_boxes", False),
        )

        self.ema = EMAModel(
            self.model,
            do_ema=config.do_ema,
            alpha=config.ema_decay,
            tau=config.ema_tau
        )

        self._freeze_layers()
        self.best_metrics: Dict[str, float] = {}
        self.best_model_path: Optional[Path] = None
        self.last_model_path: Optional[Path] = None
        self.save_dir: Optional[Path] = None
        self.weights_dir: Optional[Path] = None
        self.results_csv_path: Optional[Path] = None


        self.train_loader = None
        self.val_loader = None




    def _set_up_training(self):
        self.grid_size = max(int(self.model.model[-1].stride.max()), 32)

        self.train_loader, self.val_loader = create_dataloaders(self.config)
        self.accumulate = max(round(self.config.reference_batch_size / self.config.batch_size), 1)
        weight_decay =  self.config.weight_decay or 5e-4
        self.weight_decay = weight_decay * self.config.batch_size * self.accumulate / self.config.reference_batch_size
        self.optimizer = self._build_optimizer(weight_decay=self.weight_decay)

        self.scheduler = self._build_scheduler()
        self.early_stopping = EarlyStopping(patience=self.config.patience)

        self._setup_save_dir()
    
    def _multiscale_image_size(self, batch: dict) -> dict:
        images = batch["images"]
        multiscale_value = random.choice(range(int(self.config.imgsz * 0.5), int(self.config.imgsz * 1.5) + self.grid_size, self.grid_size))
        multiscale_ratio = multiscale_value / max(images.shape[2:])
        if multiscale_ratio != 1.0:
            new_size = [math.ceil(shape * multiscale_ratio / self.grid_size) * self.grid_size for shape in images.shape[2:]]
            images = nn.functional.interpolate(images, size=new_size, mode="bilinear", align_corners=False)
        batch["images"] = images
        return batch

    def _setup_save_dir(self) -> None:
        if not self.config.save:
            return
        project = Path(self.config.project or "runs")
        name = self.config.name or "exp"
        run_dir = project / name / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        self.save_dir = run_dir
        self.weights_dir = weights_dir
        self.results_csv_path = run_dir / "results.csv"

        try:
            if is_dataclass(self.config):
                args_dict = asdict(self.config)
            elif hasattr(self.config, "to_dict"):
                args_dict = self.config.to_dict()
            else:
                args_dict = dict(vars(self.config))
            with open(run_dir / "training_config.json", "w") as f:
                json.dump(args_dict, f, indent=2)
        except Exception as exc:
            log_info("Could not dump training_config.json: %s", exc)

        try:
            deploy = {
                "name": self.config.name,
                "img_size": self.config.imgsz,
            }
            with open(run_dir / "deployed_config.json", "w") as f:
                json.dump(deploy, f, indent=2)
        except Exception as exc:
            log_info("Could not dump deploy_config.json: %s", exc)

        log_info("Save dir: %s", run_dir)

    _CSV_COLUMNS = (
        "epoch",
        "train_loss",
        "val_loss",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "fitness",
    )

    def _append_results_csv(self, row: Dict[str, float]) -> None:
        if self.results_csv_path is None:
            return
        write_header = not self.results_csv_path.exists()
        with open(self.results_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._CSV_COLUMNS))
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._CSV_COLUMNS})

        

    def fit(self) -> Dict[str, float | int | str]:
        torch.manual_seed(self.config.seed or 0)
        self._set_up_training()
        num_batches = len(self.train_loader)
        num_warmups = max(round(self.config.warmup_epochs * num_batches), 100) if self.config.warmup_epochs > 0 else -1
        last_optimized_step = -1

        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self._scaler = scaler
        stop_early = False

        # --- Resume support ---
        start_epoch = 0
        resume_cfg = getattr(self.config, "resume", False)
        resume_path: Optional[Path] = None
        if resume_cfg:
            if isinstance(resume_cfg, (str, Path)):
                resume_path = Path(resume_cfg)
            elif self.weights_dir is not None:
                candidate = self.weights_dir / "last.pt"
                if candidate.exists():
                    resume_path = candidate
            if resume_path is None or not resume_path.exists():
                log_info("Resume requested but no checkpoint found (resume=%s). Starting fresh.", resume_cfg)
            else:
                log_info("Resuming training from %s", resume_path)
                start_epoch = self.load_checkpoint(str(resume_path), resume=True)
                # last_optimized_step ensures the very first batch after resume still steps.
                last_optimized_step = -1
                log_info("Resumed at epoch %d (best_metrics=%s)", start_epoch, self.best_metrics)

        for epoch in range(start_epoch, self.config.epochs):
            close_mosaic = self.config.close_mosaic or 0
            if close_mosaic > 0 and epoch == self.config.epochs - close_mosaic:
                log_info("Closing mosaic augmentation for the last %d epochs.", close_mosaic)
                self.train_loader.dataset.close_mosaic(
                    asdict(self.config) if is_dataclass(self.config) else vars(self.config)
                )

            running_loss = None
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            #self.scheduler.step()
            accum = self.accumulate
            pbar = tqdm(
                enumerate(self.train_loader),
                total=num_batches,
                desc=f"Epoch {epoch+1}/{self.config.epochs}",
                unit="batch",
            )
            for step, batch in pbar:
                if self.config.multi_scale:
                    batch = self._multiscale_image_size(batch)
                images = batch["images"].to(self.device, non_blocking=True)
                targets = batch["targets"].to(self.device, non_blocking=True)
                # Keep loss's img_size in sync with the actual input H/W (matters when multi_scale changes it).
                self.loss_fn.img_size = torch.tensor(images.shape[2:], dtype=torch.float, device=self.device)

                num_iterations = step + num_batches * epoch
                if num_iterations <= num_warmups:
                    progress = min(num_iterations / num_warmups, 1)
                    accum = max(int(round(1 + progress * (self.accumulate - 1))), 1)

                    for opt_index,opt_value in enumerate(self.optimizer.param_groups):
                        initial_warmup_lr = self.config.warmup_bias_lr if opt_index == 0 else 0.0
                        initial_lr = opt_value["initial_lr"] * self.lf(epoch)
                        opt_value["lr"] = initial_warmup_lr + progress * (initial_lr - initial_warmup_lr)

                        if "momentum" in opt_value:
                            opt_value["momentum"] = self.config.warmup_momentum + progress * (self.config.momentum - self.config.warmup_momentum)

                
                with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    outputs = self.model(images)
                    loss, loss_items = self.loss_fn(outputs, targets)

                scaler.scale(loss).backward()
                should_step = num_iterations - last_optimized_step >= accum
                if should_step:
                    last_optimized_step = num_iterations
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.update(self.model)

                running_loss = (running_loss * step + loss_items) / (step + 1) if running_loss is not None else loss_items
                pbar.set_postfix(
                    loss=f"{float(running_loss.sum()):.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

            self.ema.update_attribute(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            avg_train_loss = running_loss #running_loss / max(1, len(self.train_loader))

            if self.val_loader is not None and epoch >= self.config.warmup_epochs:
                fitness, val_metrics = self.validate(self.config, self.val_loader)
            else:
                fitness, val_metrics = 0.0, {}
            stop_early = self.early_stopping(fitness, epoch)
            self.scheduler.step()

            train_loss_scalar = float(avg_train_loss.sum()) if isinstance(avg_train_loss, torch.Tensor) else float(avg_train_loss or 0.0)
            current_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss_scalar,
                **val_metrics,
            }

            if epoch >= self.config.warmup_epochs:
                log_info(
                    "Epoch %d/%d - train_loss: %.4f val_loss: %.4f P: %.4f R: %.4f mAP50: %.4f mAP50-95: %.4f fitness: %.4f",
                    epoch + 1,
                    self.config.epochs,
                    train_loss_scalar,
                    current_metrics.get("val_loss", 0.0),
                    current_metrics.get("precision", 0.0),
                    current_metrics.get("recall", 0.0),
                    current_metrics.get("mAP50", 0.0),
                    current_metrics.get("mAP50-95", 0.0),
                    current_metrics.get("fitness", 0.0),
                )
            else:
                log_info(
                    "Epoch %d/%d - train_loss: %.4f (warmup)",
                    epoch + 1,
                    self.config.epochs,
                    train_loss_scalar,
                )

            if self.config.save:
                self._append_results_csv(current_metrics)
                self.last_model_path = self._save_checkpoint(epoch, filename="last.pt")

            if self._is_improved(current_metrics):
                self.best_metrics = current_metrics
                if self.config.save:
                    self.best_model_path = self._save_checkpoint(epoch, filename="best.pt")

            if stop_early:
                log_info("Early stopping triggered. No improvement in fitness for %d epochs.", self.early_stopping.patience)
                break

        # Final eval: reload best checkpoint and run validation..
        if self.best_model_path is not None and self.val_loader is not None:
            try:
                self.load_checkpoint(str(self.best_model_path))
                final_fitness, final_metrics = self.validate(self.config, self.val_loader)
                log_info("Final eval on best.pt - fitness: %.4f mAP50: %.4f mAP50-95: %.4f",
                         final_fitness,
                         final_metrics.get("mAP50", 0.0),
                         final_metrics.get("mAP50-95", 0.0))
                self.best_metrics = {**self.best_metrics, **final_metrics, "fitness": final_fitness}
                self._log_per_class_metrics(final_metrics)
            except Exception as exc:
                log_info("Final eval skipped: %s", exc)

        # export best.pt to onnx.
        if self.config.export_onnx and self.best_model_path is not None:
            try:
                _export_onnx(
                    self.best_model_path,
                    imgsz=self.config.imgsz,
                    opset=self.config.export_opset,
                    dynamic=self.config.export_dynamic,
                    simplify=self.config.export_simplify,
                    use_ema=self.config.export_ema,
                )
            except Exception as exc:
                log_info("onnx export failed: %s", exc)

        # Write summary stats file
        if self.save_dir is not None:
            try:
                summary = {
                    "epochs_trained": int(self.best_metrics.get("epoch", self.config.epochs)),
                    "best": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in self.best_metrics.items() if isinstance(v, (int, float))},
                    "best_model": str(self.best_model_path) if self.best_model_path else None,
                    "last_model": str(self.last_model_path) if self.last_model_path else None,
                }
                with open(self.save_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
            except Exception as exc:
                log_info("Could not write summary.json: %s", exc)

        result: Dict[str, float | int | str] = {
            "epochs": self.config.epochs,
            "train_loss": self.best_metrics.get("train_loss", 0.0),
        }
        if "val_loss" in self.best_metrics:
            result["val_loss"] = self.best_metrics["val_loss"]
        if self.best_model_path is not None:
            result["best_model"] = str(self.best_model_path)
        return result

    def validate(self,
                 config: SimplifiedConfig | None = None,
                 val_loader=None) -> Tuple[float, Dict[str, float]]:
        if config is None:
            config = self.config
        if val_loader is None:
            val_loader = self.val_loader
        if val_loader is None:
            try:
                _, val_loader = create_dataloaders(self.config)
                self.val_loader = val_loader
            except Exception:
                return 0.0, {}

        
        if self.ema.do_ema:
            model = self.ema.ema_model
        else:
            model = self.model     

        was_training = model.training
        model.eval()
        total_loss = 0.0
        # [box, cls, dfl]
        loss_items_sum = torch.zeros(3)  
        image_counter = 0
        all_preds = []
        all_targets = []
        with torch.inference_mode():
            for step, batch in tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc="Validating",
                unit="batch",
            ):
                
                images = batch["images"].to(self.device)
                targets = batch["targets"].to(self.device)
                self.loss_fn.img_size = torch.tensor(images.shape[2:], dtype=torch.float, device=self.device)
                
                with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    outputs = model(images)
                    raw_outputs = outputs[1] if isinstance(outputs, (tuple, list)) else outputs
                    _, loss_items = self.loss_fn(raw_outputs, targets)

                decoded = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                loss_items_cpu = loss_items.detach().cpu().float()
                loss_items_sum += loss_items_cpu
                total_loss += float(loss_items_cpu.sum())


                img_shape = images.shape[2:]
                targets_xyxy = decode_bboxes(targets, img_shape)
                single_class = getattr(config, "single_class", False)
                classes_to_ignore = getattr(config, "classes_to_ignore", None) or []
                nms_out = nms(decoded, conf_thres=0.001, iou_thres=0.7, multi_label=True, max_det=300,max_nms=30000)
                for batch_idx, pred in enumerate(nms_out):
                    pred = pred.detach().cpu()
                    if len(pred) > 0:
                        if single_class:
                            pred[:, 5] = 0
                        if classes_to_ignore:
                            keep = ~torch.isin(pred[:, 5].long(), torch.tensor(classes_to_ignore, dtype=torch.long))
                            pred = pred[keep]
                    #if len(pred) > 0:
                        image_counters = torch.full((pred.shape[0], 1), image_counter, dtype=pred.dtype)
                        processed = torch.cat([image_counters, pred[:, 5:6], pred[:, 4:5], pred[:, :4]], dim=1)
                        all_preds.append(processed)

                    targ_indices = torch.where(targets_xyxy[:, 0] == batch_idx)[0]
                    targs = targets_xyxy[targ_indices].detach().cpu()
                    if len(targs) > 0:
                        image_counters = torch.full((targs.shape[0], 1), image_counter, dtype=targs.dtype)
                        processed_targ = torch.cat([image_counters, targs[:, 1:2], targs[:, 2:]], dim=1)
                        all_targets.append(processed_targ)
                    image_counter += 1
        all_preds = torch.cat(all_preds, dim = 0) if len(all_preds) > 0 else torch.empty((0, 7))
        all_targets = torch.cat(all_targets, dim = 0) if len(all_targets) > 0 else torch.empty((0, 6))
        map_50_metrics, map_50_95_metrics, fitness = mean_avg_precision_metric(
            all_preds,
            all_targets,
            num_classes=getattr(self.model, "nc", config.extra.get("nc", 80)),
        )
        nb = max(1, len(val_loader))
        self.last_class_map = {
            "map_50": map_50_metrics.get("class_map", []),
            "map_50_95": map_50_95_metrics.get("class_map", []),
        }
        metrics: Dict[str, float] = {
            "val_loss": total_loss / nb,
            "val_box_loss": float(loss_items_sum[0]) / nb,
            "val_cls_loss": float(loss_items_sum[1]) / nb,
            "val_dfl_loss": float(loss_items_sum[2]) / nb,
            "precision": float(map_50_metrics.get("precision", 0.0)),
            "recall": float(map_50_metrics.get("recall", 0.0)),
            "mAP50": float(map_50_metrics.get("map", 0.0)),
            "mAP50-95": float(map_50_95_metrics.get("map", 0.0)),
            "fitness": float(fitness),
        }

        if was_training:
            model.train()
        del all_preds, all_targets, nms_out, decoded
        torch.cuda.empty_cache()
        return float(fitness), metrics

    def _log_per_class_metrics(self, final_metrics: Dict[str, float]) -> None:
        """Log overall + per-class P / R / mAP50 / mAP50-95 and persist to JSON."""
        per_class_50 = (getattr(self, "last_class_map", {}) or {}).get("map_50", []) or []
        per_class_95 = (getattr(self, "last_class_map", {}) or {}).get("map_50_95", []) or []
        names = getattr(self.model, "names", None) or getattr(self.ema.ema_model, "names", None)

        def _name(idx: int) -> str:
            if isinstance(names, dict):
                return str(names.get(idx, idx))
            if isinstance(names, (list, tuple)) and 0 <= idx < len(names):
                return str(names[idx])
            return str(idx)

        header = f"{'Class':<20} {'P':>10} {'R':>10} {'mAP50':>10} {'mAP50-95':>10}"
        log_info("Final per-class metrics on best.pt")
        log_info(header)
        log_info("-" * len(header))
        log_info(
            f"{'all':<20} "
            f"{final_metrics.get('precision', 0.0):>10.4f} "
            f"{final_metrics.get('recall', 0.0):>10.4f} "
            f"{final_metrics.get('mAP50', 0.0):>10.4f} "
            f"{final_metrics.get('mAP50-95', 0.0):>10.4f}"
        )

        by_class_95 = {e["class"]: e for e in per_class_95}
        per_class_rows = []
        for entry in per_class_50:
            cls_idx = entry["class"]
            e95 = by_class_95.get(cls_idx, {})
            row = {
                "class_idx": int(cls_idx),
                "class_name": _name(int(cls_idx)),
                "precision": float(entry.get("precisions", 0.0)),
                "recall": float(entry.get("recalls", 0.0)),
                "mAP50": float(entry.get("AP", 0.0)),
                "mAP50-95": float(e95.get("AP", 0.0)),
            }
            per_class_rows.append(row)
            log_info(
                f"{row['class_name']:<20} "
                f"{row['precision']:>10.4f} "
                f"{row['recall']:>10.4f} "
                f"{row['mAP50']:>10.4f} "
                f"{row['mAP50-95']:>10.4f}"
            )

        if self.save_dir is not None:
            try:
                payload = {
                    "all": {
                        "precision": float(final_metrics.get("precision", 0.0)),
                        "recall": float(final_metrics.get("recall", 0.0)),
                        "mAP50": float(final_metrics.get("mAP50", 0.0)),
                        "mAP50-95": float(final_metrics.get("mAP50-95", 0.0)),
                        "fitness": float(final_metrics.get("fitness", 0.0)),
                    },
                    "per_class": per_class_rows,
                }
                with open(self.save_dir / "final_metrics.json", "w") as f:
                    json.dump(payload, f, indent=2)
                if per_class_rows:
                    csv_path = self.save_dir / "final_metrics_per_class.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["class_idx", "class_name", "precision", "recall", "mAP50", "mAP50-95"],
                        )
                        writer.writeheader()
                        writer.writerows(per_class_rows)
            except Exception as exc:
                log_info("Could not write final per-class metrics: %s", exc)

    def _build_optimizer(self,
                         weight_decay: float = 5e-4) -> torch.optim.Optimizer:
        lr = self.config.lr0 or 1e-3

        optimizer_name = (self.config.optimizer or "adamw").lower()
        
        g_decay, g_no_decay, g_bias = [], [], []

        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                      nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, 
                      nn.InstanceNorm2d, nn.InstanceNorm3d)
        
        for module in self.model.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if "bias" in param_name:
                    g_bias.append(param)
                elif isinstance(module, norm_types):
                    g_no_decay.append(param)
                else:
                    g_decay.append(param)
        
        if optimizer_name == "sgd":
            optimizer = SGD(g_bias, 
                            lr=lr, 
                            momentum=self.config.momentum, 
                            nesterov=True)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(g_bias, 
                                         lr=lr, 
                                         betas=(self.config.momentum, 0.999))
        else:
            optimizer = AdamW(g_bias,
                              betas=(self.config.momentum, 0.999), 
                              lr=lr)
        
        optimizer.add_param_group({"params": g_decay, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": g_no_decay, "weight_decay": 0.0})
        
        log_info("Optimizer: %s with %d bias, %d weight(decay=%.4f), %d norm(decay=0.0) params",
                 optimizer_name, len(g_bias), len(g_decay), weight_decay, len(g_no_decay))
        return optimizer

    def _build_scheduler(self):
        lrf = self.config.lrf or 0.01
        #lr0 = self.config.lr0
        if self.config.epochs <= 0:
            return None

        def lf(x: int) -> float:
            return (max((1 - x / self.config.epochs),0) * (1 - lrf) + lrf)
        
        def coslr(x: int) -> float:
            return max(((1 - math.cos( (x * math.pi) / self.config.epochs)) / 2 ), 0) *  (lrf - 1) + 1
        
        if self.config.cos_lr:
            self.lf = coslr
        else:
            self.lf = lf

        return LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _is_improved(self, metrics: Dict[str, float]) -> bool:
        if not self.best_metrics:
            return True
        # Higher fitness is better; fall back to val_loss (lower is better) during warmup.
        if "fitness" in metrics and "fitness" in self.best_metrics:
            return metrics["fitness"] > self.best_metrics["fitness"]
        return metrics.get("val_loss", math.inf) < self.best_metrics.get("val_loss", math.inf)

    def _save_checkpoint(self, epoch: int, filename: str = "best.pt") -> Path:
        if self.weights_dir is None:
            self._setup_save_dir()
        weights_dir = self.weights_dir or (Path(self.config.project or "runs") / (self.config.name or "exp") / "train" / "weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = weights_dir / filename
        names = {}
        try:
            data_path = Path(self.config.data)
            if data_path.exists():
                with data_path.open() as f:
                    d = yaml.safe_load(f) or {}
                raw = d.get("names", {})
                if isinstance(raw, dict):
                    names = {int(k): str(v) for k, v in raw.items()}
                elif isinstance(raw, (list, tuple)):
                    names = {i: str(n) for i, n in enumerate(raw)}
        except Exception:
            pass
        payload = {
            "epoch": epoch + 1,
            "model": self.model.state_dict(),
            "ema": self.ema.ema_model.state_dict(),
            "ema_updates": self.ema.update_after_step,
            "config": relativize_config_paths(self.config),
            "names": names,
        }
        # Reload optimizer, scheduler, AMP scaler, RNG, best metrics.
        if getattr(self, "optimizer", None) is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        if getattr(self, "scheduler", None) is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        if getattr(self, "_scaler", None) is not None:
            payload["scaler"] = self._scaler.state_dict()
        payload["best_metrics"] = self.best_metrics
        payload["torch_rng"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(payload, checkpoint_path)
        log_info("Saved checkpoint to %s", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, resume: bool = False) -> int:
        # Trainer checkpoints embed a SimplifiedConfig dataclass, so weights_only=True
        # cannot deserialize them.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            self.ema.ema_model.load_state_dict(ckpt["ema"])
            self.ema.update_after_step = ckpt.get("ema_updates", 0)
        if resume:
            if "optimizer" in ckpt and getattr(self, "optimizer", None) is not None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and getattr(self, "scheduler", None) is not None:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and getattr(self, "_scaler", None) is not None:
                self._scaler.load_state_dict(ckpt["scaler"])
            if "best_metrics" in ckpt:
                self.best_metrics = ckpt["best_metrics"] or {}
            if "torch_rng" in ckpt:
                torch.set_rng_state(ckpt["torch_rng"].cpu())
            if "cuda_rng" in ckpt and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
                except Exception:
                    pass
        epoch = ckpt.get("epoch", 0)
        del ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return epoch
    
    def _freeze_layers(self):
        if not isinstance(self.config.freeze, int) or self.config.freeze < 0:
            log_info("Invalid freeze value '%s'. Expected a non-negative integer. No layers will be frozen.", self.config.freeze)
            return
        if self.config.freeze == 0:
            for name, param in self.model.named_parameters():
                if ".dfl" in name:
                    param.requires_grad = False
            return
        
        freeze_list = [i for i in range(self.config.freeze)]
        always_freeze = ".dfl"
        freeze_layers = [f"model.{layer_number}." for layer_number in freeze_list]
        freeze_layers.append(always_freeze)
        #freeze_layers += always_freeze

        for name, param in self.model.named_parameters():
            name_split = name.split(".")
            name_layer = name_split[0] + "." + name_split[1] +"."
            if name_layer in freeze_layers: #any(name.startswith(layer) for layer in freeze_layers):
                param.requires_grad = False
                log_info("Freezing layer: %s", name)
            elif param.requires_grad == False and param.dtype.is_floating_point:
                param.requires_grad = True
                log_info("Unfreezing layer: %s", name)

            if always_freeze in name:
                param.requires_grad = False

    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        if preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        try:
            return torch.device(preference)
        except RuntimeError:
            log_info("Falling back to CPU device. Requested device '%s' unavailable.", preference)
            return torch.device("cpu")
