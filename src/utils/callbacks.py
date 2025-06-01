import subprocess
import warnings
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callbacks import Callback
from omegaconf import DictConfig


class GitSHACallback(Callback):
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        hydra_folder = HydraConfig.get().runtime.output_dir
        commit_sha_file = Path(hydra_folder) / "sha.txt"
        with open(commit_sha_file, "w") as f:
            try:
                f.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8"))
            except subprocess.CalledProcessError:
                warnings.warn("Could not get git commit sha.")
