import json
import warnings
from pathlib import Path

import torch
import yaml

from ..core.logger import log_info
from ..core.model import SimplifiedDetectionModel
from ..core.modules.detect import Detect


def resolve_model_cfg(cfg_path: str | Path) -> str:
    p = Path(cfg_path)
    if p.exists():
        return str(p)
    if p.is_absolute():
        # Try progressively shorter suffixes relative to cwd
        parts = p.parts[1:]  
        for i in range(len(parts)):
            candidate = Path(*parts[i:])
            if candidate.exists():
                log_info("resolved model cfg '%s' -> '%s'", cfg_path, candidate)
                return str(candidate)
    raise FileNotFoundError(
        f"Model config not found: '{cfg_path}'. "
        "Pass an explicit model_cfg path or ensure the file exists relative to cwd."
    )


def get_names(config, nc):
    # checkpoint-embedded names take priority (set by load_pt_model / export_onnx)
    ckpt_names = getattr(config, "_ckpt_names", None)
    if ckpt_names:
        return ckpt_names
    # fall back to the dataset yaml
    if config.data:
        p = Path(config.data)
        if p.exists():
            with p.open() as f:
                d = yaml.safe_load(f) or {}
            names = d.get("names")
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, (list, tuple)):
                return {i: str(n) for i, n in enumerate(names)}
    return {i: str(i) for i in range(nc)}

def load_pt_model(checkpoint,
                  imgsz=None,
                  use_ema=False,
                  model_cfg=None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = ckpt.get("config")
    if config is None:
        raise ValueError(f"no 'config' in {checkpoint}, can't rebuild the model")
    if imgsz is None:
        imgsz = int(config.imgsz)
    nc_override = config.extra.get("nc")
    cfg_path = model_cfg if model_cfg is not None else config.model
    model = SimplifiedDetectionModel(cfg=resolve_model_cfg(cfg_path), number_of_classes=nc_override)
    key = "ema" if use_ema else "model"
    if key not in ckpt:
        raise KeyError(f"{key} weights not in checkpoint (got {list(ckpt)})")
    missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
    if missing:
        log_info("missing keys: %s", missing)
    if unexpected:
        log_info("unexpected keys: %s", unexpected)
    ckpt_names = ckpt.get("names")
    if ckpt_names:
        config._ckpt_names = {int(k): str(v) for k, v in ckpt_names.items()}
    return model, config, imgsz

def export_onnx(
    checkpoint,
    output=None,
    imgsz=None,
    opset=12,
    dynamic=False,
    batch=1,
    simplify=False,
    fuse=True,
    use_ema=False,
    device="cpu",
    model_cfg=None,
):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    # Trainer checkpoints embed a SimplifiedConfig dataclass under "config", so
    # weights_only=True cannot deserialize them. These files are produced by us
    # and are trusted.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`",
            category=FutureWarning,
        )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    if config is None:
        raise ValueError(f"no 'config' in {checkpoint}, can't rebuild the model")

    if imgsz is None:
        imgsz = int(config.imgsz)
    if output is None:
        output = checkpoint.with_suffix(".onnx")
    output = Path(output)

    nc_override = config.extra.get("nc")
    cfg_path = model_cfg if model_cfg is not None else config.model
    model = SimplifiedDetectionModel(cfg=resolve_model_cfg(cfg_path), number_of_classes=nc_override)

    key = "ema" if use_ema else "model"
    if key not in ckpt:
        raise KeyError(f"{key} weights not in checkpoint (got {list(ckpt)})")
    missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
    if missing:
        log_info("missing keys: %s", missing)
    if unexpected:
        log_info("unexpected keys: %s", unexpected)
    ckpt_names = ckpt.get("names")
    if ckpt_names:
        config._ckpt_names = {int(k): str(v) for k, v in ckpt_names.items()}

    # tracing dynamic shapes on cuda is flaky
    if dynamic:
        device = "cpu"  
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    model.eval().float()
    if fuse:
        model.fuse()

    detect = None
    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
            m.dynamic = bool(dynamic)

            detect = m

    nc = int(getattr(model, "number_of_classes", None) or config.extra.get("nc", 80))
    names = get_names(config, nc)

    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    # warm up the anchor cache before tracing
    with torch.no_grad():
        model(dummy)
        model(dummy)

    axes = None
    if dynamic:
        axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "output0": {0: "batch", 2: "anchors"},
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            str(output),
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes=axes,
        )

    stride = detect.stride.tolist() if detect is not None else [8, 16, 32]
    meta = {
        "task": "detect",
        "nc": nc,
        "names": names,
        "imgsz": [imgsz, imgsz],
        "stride": stride,
        "batch": batch,
        "weights": key,
    }
    write_metadata(output, meta)

    if simplify:
        run_simplify(output)

    log_info("onnx written to %s (opset=%d dynamic=%s weights=%s)", output, opset, dynamic, key)
    return output


def write_metadata(path, meta):
    try:
        import onnx
    except ImportError:
        return
    m = onnx.load(str(path))
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key = str(k)
        e.value = v if isinstance(v, str) else json.dumps(v)
    onnx.save(m, str(path))


def run_simplify(path):
    try:
        import onnx
    except ImportError:
        log_info("onnx not installed, can't simplify")
        return

    try:
        import onnxslim
        m = onnxslim.slim(str(path))
    except ImportError:
        try:
            from onnxsim import simplify
        except ImportError:
            log_info("neither onnxslim nor onnxsim installed, skipping simplify")
            return
        m = onnx.load(str(path))
        m, ok = simplify(m)
        if not ok:
            log_info("onnxsim returned failure, keeping original")
            return

    onnx.save(m, str(path))
    log_info("simplified %s", path)
