""" file with CLI to run training"""
from pytorch_lightning.utilities.cli import LightningCLI

from experiment.sentiment_datamodule import SentimentDatamodule
from experiment.sentiment_model import SentimentModel


class SentimentExperimentCLI(LightningCLI):
    def fit(self):
        self.trainer.fit(**self.fit_kwargs)


if __name__ == "__main__":
    cli = SentimentExperimentCLI(
        SentimentModel,
        SentimentDatamodule,
        save_config_overwrite=True,
    )
