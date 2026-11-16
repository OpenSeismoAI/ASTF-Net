from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.data_io.dataset import SeismicDatasetHDF5
from astfnet.models.base import ASTFModule


@pytest.fixture
def dummy_config(tmp_path: Path) -> Dict[str, Any]:
    # Create a dummy HDF5 file for testing
    h5_path = tmp_path / "dummy.h5"
    n, length = 10, 100
    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("target_waveforms", data=rng.normal(size=(n, length)))
        f.create_dataset("egfs", data=rng.normal(size=(n, length)))
        f.create_dataset("astfs", data=rng.normal(size=(n, length)))

    config = {
        "train_hdf5_file": str(h5_path),
        "val_hdf5_file": str(h5_path),
        "test_hdf5_file": str(h5_path),
        "batch_size": 2,
        "num_workers": 0,
        "data_augmentations": [
            {"AddNoise": {"snr_low": 10, "snr_high": 20}},
            {"DropChunk": {"drop_length_low": 5, "drop_length_high": 10}},
        ],
    }
    return config


def test_seismic_dataset_with_augmentation(dummy_config: Dict[str, Any]) -> None:
    dataset = SeismicDatasetHDF5(
        dummy_config["train_hdf5_file"],
        {"data_augmentations": dummy_config["data_augmentations"]},
    )
    sample = dataset[0]
    assert "target" in sample and "egf" in sample and "astf" in sample
    assert isinstance(sample["target"], torch.Tensor)
    assert isinstance(sample["egf"], torch.Tensor)
    assert isinstance(sample["astf"], torch.Tensor)
    # Check that augmentation does not change shape
    assert sample["target"].shape == torch.Size([100])
    assert sample["egf"].shape == torch.Size([100])
    assert sample["astf"].shape == torch.Size([100])


def test_seismic_dataset_without_augmentation(dummy_config: Dict[str, Any]) -> None:
    dataset = SeismicDatasetHDF5(dummy_config["val_hdf5_file"])
    sample = dataset[0]
    assert "target" in sample and "egf" in sample and "astf" in sample
    assert isinstance(sample["target"], torch.Tensor)
    assert isinstance(sample["egf"], torch.Tensor)
    assert isinstance(sample["astf"], torch.Tensor)
    assert sample["target"].shape == torch.Size([100])
    assert sample["egf"].shape == torch.Size([100])
    assert sample["astf"].shape == torch.Size([100])


@pytest.mark.parametrize("num_workers", [0, 1])
def test_seismic_datamodule(dummy_config: Dict[str, Any], num_workers: int) -> None:
    dm = SeismicDataModule(
        train_hdf5_file=dummy_config["train_hdf5_file"],
        val_hdf5_file=dummy_config["val_hdf5_file"],
        test_hdf5_files=dummy_config["test_hdf5_file"],
        batch_size=dummy_config["batch_size"],
        num_workers=num_workers,
    )
    dm.setup("fit")
    dm.setup("test")
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        batches = list(loader)
        assert len(batches) == 5
        for batch in batches:
            assert set(batch) == {"target", "egf", "astf"}
            for tensor in batch.values():
                assert tensor.shape == torch.Size([2, 100])
                assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("num_workers", [0, 1])
def test_lightning_training_with_hdf5(tmp_path: Path, num_workers: int) -> None:
    h5_path = tmp_path / "training.h5"
    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("target_waveforms", data=rng.normal(size=(4, 256)))
        f.create_dataset("egfs", data=rng.normal(size=(4, 256)))
        f.create_dataset("astfs", data=np.abs(rng.normal(size=(4, 100))))

    dm = SeismicDataModule(str(h5_path), str(h5_path), batch_size=2, num_workers=num_workers)
    torch.manual_seed(42)
    model = ASTFModule({"model_name": "simplecnn", "in_channels": 2, "output_length": 100, "loss": "mse"})
    parameter = next(model.parameters())
    initial_parameter = parameter.detach().clone()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, datamodule=dm)
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["train/loss_epoch"])
    assert torch.isfinite(trainer.callback_metrics["val/loss_epoch"])
    assert not torch.equal(parameter.detach(), initial_parameter)


def test_dataloader(dummy_config: Dict[str, Any]) -> None:
    dataset = SeismicDatasetHDF5(dummy_config["test_hdf5_file"])
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    assert "target" in batch and "egf" in batch and "astf" in batch
