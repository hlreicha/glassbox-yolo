from __future__ import annotations
import os

import argparse
import json
from typing import Any, Dict

from yolo.engine.export import export_onnx as _export_onnx

def read_json(path):
    f = open(path)
    data = json.load(f)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser("config_file")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file")
    args = parser.parse_args()
    config = read_json(args.config)


    root_path = os.path.join(config["project"],config["name"],"train")
    weight_path = os.path.join(root_path,"weights","best.pt")

    _export_onnx(
        weight_path,
        imgsz= config["imgsz"],
        opset=config["export_opset"],
        dynamic=config["export_dynamic"],
        simplify=config["export_simplify"],
        use_ema=config["export_ema"],
    )