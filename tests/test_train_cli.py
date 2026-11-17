import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from astfnet.cli import train


@pytest.mark.parametrize("explicit_config", [False, True])
def test_train_cli_config_from_other_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_config: bool
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    example_config = OmegaConf.load(project_root / "config" / "config-simplecnn.yaml")
    command = ["astfnet-train", "--data", str(tmp_path), "--skip-test"]
    expected_epochs = example_config.max_epochs
    if explicit_config:
        example_config.max_epochs = 1
        OmegaConf.save(example_config, tmp_path / "custom.yaml")
        command.extend(["--config", "custom.yaml"])
        expected_epochs = 1

    trainer_factory = Mock()
    monkeypatch.setattr(train.pl, "Trainer", trainer_factory)
    monkeypatch.setattr(train, "TensorBoardLogger", Mock())
    monkeypatch.setattr(sys, "argv", command)
    monkeypatch.chdir(tmp_path)
    train.main()

    trainer_factory.return_value.fit.assert_called_once()
    model = trainer_factory.return_value.fit.call_args.args[0]
    assert model.hparams.model_name == "simplecnn"
    assert trainer_factory.call_args.kwargs["max_epochs"] == expected_epochs
    datamodule = trainer_factory.return_value.fit.call_args.kwargs["datamodule"]
    assert Path(datamodule.train_hdf5_file).parent == tmp_path
    trainer_factory.return_value.test.assert_not_called()


def test_bundled_config_matches_example() -> None:
    package_dir = Path(train.__file__).resolve().parents[1]
    example = Path(__file__).resolve().parents[1] / "config" / "config-simplecnn.yaml"
    assert (package_dir / "config-simplecnn.yaml").read_bytes() == example.read_bytes()


@pytest.mark.parametrize("config_name", ["missing.yaml", "."])
def test_train_cli_missing_config(tmp_path: Path, config_name: str) -> None:
    config_path = tmp_path / config_name
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("astfnet-train")),
            "--config",
            str(config_path),
            "--data",
            str(tmp_path),
            "--skip-test",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert f"Config file not found: {config_path}" in result.stderr
    assert "Traceback" not in result.stderr
