import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
# from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, LightningModule
from utils import instantiate_callbacks, instantiate_loggers

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="./configs", config_name="config_IText")
def main(cfg: DictConfig):
    # log.info(OmegaConf.to_yaml(cfg))
    callbacks = instantiate_callbacks(cfg.callbacks)

    loggers = instantiate_loggers(cfg.loggers)

    datamodule: LightningDataModule = instantiate(cfg.datamodule)

    trainer: Trainer = instantiate(cfg.trainer)
    trainer = trainer(
        callbacks=callbacks,
        logger=loggers,
    )

    lit_module: LightningModule = instantiate(cfg.module)
    
    trainer.fit(
        model=lit_module,
        datamodule=datamodule,
        # ckpt_path=cfg.ckpt_path,
    )

if __name__ == "__main__":
    main()
