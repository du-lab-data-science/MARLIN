"""Train the MARLIN decoder (Hydra config: configs/train_marlin_decoder.yaml).

    python -m marlin.train
    python -m marlin.train trainer.max_steps=1 loader.global_batch_size=8     # quick check
"""

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# make the packages in src/ importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import hydra
import lightning as L
import omegaconf
import torch

from marlin.model import MarlinDecoder
from marlin.data import get_msbd3_dataloader
from dlm.utils.utils_data import get_last_checkpoint

for name, fn in [
    ("cwd", os.getcwd),
    ("device_count", torch.cuda.device_count),
    ("eval", eval),
    ("div_up", lambda x, y: (x + y - 1) // y),
    ("home", lambda: os.path.expanduser("~")),
]:
    try:
        omegaconf.OmegaConf.register_new_resolver(name, fn)
    except Exception:
        pass


def _load_weights_only(model, checkpoint_path, strict=False):
    print(f"[msbd3-train] Loading weights only from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"[msbd3-train] missing={len(missing)} unexpected={len(unexpected)} (strict={strict})")
    return model


@hydra.main(version_base=None, config_path="configs", config_name="train_marlin_decoder")
def train(config):
    wandb_logger = None
    if config.get("wandb", {}).get("name", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    model = MarlinDecoder(config)

    # warm start from FRIGID's decoder unless a run is being resumed
    init_from_dlm = config.get("init_from_dlm", None)
    if init_from_dlm is not None and get_last_checkpoint(config.callback.dirpath) is None:
        from marlin.warm_start import load_dlm_weights_into_msbd3

        load_dlm_weights_into_msbd3(model, init_from_dlm, verbose=True)

    weights_only_path = config.get("load_weights_only", None)
    if weights_only_path is not None:
        _load_weights_only(model, weights_only_path, strict=config.get("load_weights_strict", False))

    ckpt_path = get_last_checkpoint(config.callback.dirpath)
    resume_step = 0
    if ckpt_path is not None:
        try:
            resume_step = int(os.path.basename(ckpt_path).split(".")[0])
            print(f"[msbd3-train] Resuming from step {resume_step}")
        except Exception:
            pass
    with omegaconf.open_dict(config.loader):
        config.loader.resume_step = resume_step

    train_dataloader = get_msbd3_dataloader(config)
    callbacks = [hydra.utils.instantiate(config.callback)]

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(
            {"_target_": "lightning.pytorch.strategies.DDPStrategy", "find_unused_parameters": True}
        ),
        logger=wandb_logger,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train()
