""" file with CLI to run training"""
import os

from loguru import logger
from pytorch_lightning.utilities.cli import LightningCLI

from experiment.sentiment_datamodule import SentimentDatamodule
from experiment.sentiment_model import SentimentModel


class TemporalClosenessExperimentCLI(LightningCLI):
    def fit(self):
        if os.getenv("EVAL") or os.getenv("TEST"):
            logger.info("Starting evaluation...")
            self.trainer.test(**self.fit_kwargs)
            logger.info("...evaluation finished!")
        else:
            self.trainer.fit(**self.fit_kwargs)


if __name__ == "__main__":
    cli = TemporalClosenessExperimentCLI(
        SentimentModel,
        SentimentDatamodule,
        save_config_overwrite=True,
    )
