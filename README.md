# Glassbox YOLO

A YOLO training pipeline re-implemented from scratch, mainly to understand and control every part of the process instead of relying on a large framework. It also sidesteps the AGPL-3.0 license, which requires open-sourcing any derivative or networked use of their code.

## Models

The model architecture is defined by a YAML config. The repo ships with YOLOv11 and YOLOv12 style configs, supporting the standard n/s/m/l/x scales. Pretrained weights can be loaded as a starting point.

## Augmentation

Controlled per-run via the train config:

| Parameter | Description |
|---|---|
| `mosaic` | Mosaic probability (0.0–1.0) |
| `hsv_h/s/v` | HSV color jitter |
| `fliplr` / `flipud` | Horizontal / vertical flip probability |
| `translate` | Translation fraction |
| `scale` | Scale jitter |
| `degrees` | Rotation degrees |
| `shear` | Shear degrees |
| `perspective` | Perspective distortion |
| `close_mosaic` | Disable mosaic for last N epochs |
| `rect` | Rectangular training |
| `do_letterbox` | Letterbox resizing |

## Features

- EMA (exponential moving average) of model weights
- Cosine LR schedule or flat LR decay
- Gradient accumulation
- Auto batch-size scaling via `reference_batch_size`
- Early stopping via `patience`
- Resume from `last.pt` or a specific checkpoint
- Layer freezing via `freeze`
- Multi-scale training via `multi_scale`
- Dataset filtering: `classes_to_ignore`, `single_class`
- ONNX export at end of training (optional)

## Training



Train configs are JSON or YAML files. See `train_configs/` for examples.

```bash
python -m yolo.train --config train_configs/my_config.json
```

CLI overrides:

```bash
python -m yolo.train --config train_configs/my_config.json \
    --weights trained_models/pretrain_models/pretrained_yolo11n/train/weights/best.pt \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --device cuda:0 \
    --name my_run
```

Resume training:

```bash
# auto-detect last.pt
python -m yolo.train --config train_configs/my_config.json --resume

# explicit checkpoint
python -m yolo.train --config train_configs/my_config.json --resume runs/my_project/my_run/train/weights/last.pt
```

### Pretrained YOLO11 weights

YOLO11n and YOLO11s has been pretrained so far. `trained_models/` is not tracked in git, so download the checkpoints from the [pretrained-weights-v1 release]() and place them at the paths below. Each pretrained checkpoint needs two things specified in the training config:
- The model config YAML (`yolo/model_config/yolo11.yaml`), with `scale` set to match (`n` or `s`)
- A checkpoint (`train/weights/best.pt`)

| Model | Model config | Download | Place at |
|---|---|---|---|
| YOLO11n scratch | `yolo/model_config/yolo11.yaml` (`scale: n`) | [yolo11n-best.pt]() | `trained_models/pretrain_models/pretrained_yolo11n/train/weights/best.pt` |
| YOLO11s scratch | `yolo/model_config/yolo11.yaml` (`scale: s`) | [yolo11s-best.pt](https://github.com/hlreicha/glassbox-yolo/releases/download/pretrained-weights-v1/yolo11s-best.pt) | `trained_models/pretrain_models/pretrained_yolo11s/train/weights/best.pt` |

ONNX exports of both are also on the release page ([yolo11n-best.onnx](https://github.com/hlreicha/glassbox-yolo/releases/download/pretrained-weights-v1/yolo11n-best.onnx), [yolo11s-best.onnx](https://github.com/hlreicha/glassbox-yolo/releases/download/pretrained-weights-v1/yolo11s-best.onnx)) if you just want to run inference without training.

Example config for starting from YOLO11n:

```json
{
    "model": "yolo/model_config/yolo11.yaml",
    "data": "train_configs/coco128.yaml",
    "pretrained": "trained_models/pretrain_models/pretrained_yolo11n/train/weights/best.pt",
    "project": "trained_models/my_project",
    "name": "my_yolo11n_run",
    "epochs": 100,
    "batch_size": 32,
    "imgsz": 640,
    "device": "auto",
    "extra": {
        "nc": 80,
        "reg_max": 16,
        "strides": [8, 16, 32]
    }
}
```

Then train normally:

```bash
python -m yolo.train --config train_configs/my_config.json
```

You can also override the initial weights from the CLI, but keep the `model` field matched to the same scale:

```bash
python -m yolo.train --config train_configs/my_config.json \
    --weights trained_models/pretrain_models/pretrained_yolo11s/train/weights/best.pt
```

For custom datasets, set `extra.nc` to the number of classes in your dataset. The loader copies all compatible pretrained tensors into the new head and skips or slices tensors that do not match the target class count.

### Config fields

```json
{
    "model": "yolo/model_config/yolo11.yaml",
    "data": "train_configs/coco128.yaml",
    "project": "runs/my_project",
    "name": "run1",
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
    "device": "auto",
    "pretrained": "trained_models/pretrain_models/pretrained_yolo11n/train/weights/best.pt",
    "optimizer": "sgd",
    "lr0": 0.01,
    "lrf": 0.01,
    "weight_decay": 5e-4,
    "momentum": 0.937,
    "warmup_epochs": 3.0,
    "patience": 50,
    "do_ema": true,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "export_onnx": true,
    "export_opset": 12,
    "export_dynamic": false,
    "export_simplify": false
}
```

## Export

ONNX export runs automatically at the end of training if `export_onnx: true` (default). To export manually:

```bash
python -m yolo.export_onnx --config ./trained_models/<name>/train/training_config.json
```