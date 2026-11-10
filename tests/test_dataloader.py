import sys
from dataclasses import replace

import pytest
import yaml

from yolo.data import dataset as dataset_module
from yolo.data.dataset import create_dataloaders
from yolo.engine.trainer import Trainer


def test_val_only_does_not_need_a_train_split(tiny_config, tmp_path):
    data = yaml.safe_load(open(tiny_config.data))
    del data["train"]
    val_only_yaml = tmp_path / "val_only.yaml"
    val_only_yaml.write_text(yaml.safe_dump(data))
    config = replace(tiny_config, data=str(val_only_yaml))

    train_loader, val_loader = create_dataloaders(config, val_only=True)
    assert train_loader is None
    assert len(val_loader.dataset) == 8
    with pytest.raises(KeyError):
        create_dataloaders(config)


@pytest.mark.skipif(sys.platform != "linux", reason="relies on fork-started dataloader workers")
def test_close_mosaic_reaches_persistent_workers(tiny_config, tmp_path, monkeypatch):
    flags = tmp_path / "mosaic_flags.log"
    original = dataset_module.YoloDataset.__getitem__

    def traced(self, idx):
        with open(flags, "a") as f:
            f.write(f"{self.mosaic}\n")
        return original(self, idx)

    monkeypatch.setattr(dataset_module.YoloDataset, "__getitem__", traced)
    config = replace(tiny_config, workers=2)
    train_loader, _ = create_dataloaders(config)

    def mosaic_flags_for_one_epoch():
        flags.write_text("")
        for _ in train_loader:
            pass
        return flags.read_text().split()

    assert set(mosaic_flags_for_one_epoch()) == {"True"}

    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.train_loader = train_loader
    trainer._close_mosaic()

    assert set(mosaic_flags_for_one_epoch()) == {"False"}
    assert set(mosaic_flags_for_one_epoch()) == {"False"}
