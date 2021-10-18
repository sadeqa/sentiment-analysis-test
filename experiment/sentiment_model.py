from typing import Dict, List

import torch
from loguru import logger
from pytorch_lightning import LightningModule
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from transformers import AutoModelForSequenceClassification

from utils.metrics import classification_metrics


class SentimentModel(LightningModule):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            lr: float = 0.001,
            n_classes: int = 3,
            **kwargs,
    ):
        """

        Args:
            pretrained_model_name_or_path (str): The name of the transformer backbone
            lr (float): Learning rate
            n_classes (int): The number of classes to output. Defaults to 3.
        """
        super().__init__()

        self.optimizers: List[Optimizer] = []
        self.schedulers: List = []

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lr = lr

        self.n_training_examples: int = 0
        self.metrics_dict_train, self.metrics_dict_val = classification_metrics(
            n_classes=n_classes,
        )

        self.loss = CrossEntropyLoss()
        self.process_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )
        for param in self.process_model.roberta.parameters():
            param.requires_grad = False

    def forward(
            self,
            x,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        """Function for inference without postprocess

        Args:
            input_array (Dict[str, torch.Tensor]): input tensor

        Returns:
            torch.Tensor: inferred value
        """
        out = self.process_model(
            x["input_ids"], x["attention_mask"]
        ).logits
        return out

    def postprocess_eval(self, preds: torch.Tensor) -> torch.Tensor:
        return preds

    def training_step(self, batch, *args, **kwargs):
        """Function to one training step;
            always return {"loss": loss}

        Args:
            batch (): batch input
        """
        input_value, y_target = batch
        y_pred = self.forward(input_value)
        loss = self.loss(y_pred, y_target)
        logger.debug(
            f"For training step: predict: {y_pred} vs target: {y_target}, loss: {loss}"
        )
        y_pred = self.postprocess_eval(y_pred)
        self.training_log(loss, y_pred, y_target)
        return {"loss": loss}

    def training_log(
            self,
            loss: torch.Tensor,
            y_pred: torch.Tensor,
            y_target: torch.Tensor,
    ):
        """Function to log the metrics

        Args:
            loss (torch.Tensor): The loss of that step
            y_pred (torch.Tensor): the predicted tensor
            y_target (torch.Tensor): the target tensor
        """
        if self.logger:
            self.logger.log_metrics(
                metrics={"Training/training_loss": loss.detach().item()},
                step=self.n_training_examples,
            )
            for metric_name, metric in self.metrics_dict_train.items():
                value = metric(
                    torch.argmax(y_pred, dim=1), y_target
                )
                self.logger.log_metrics(
                    metrics={f"Training/{metric_name}_step": value.item()},
                    step=self.n_training_examples,
                )

        self.n_training_examples += 1

    def validation_step(self, batch, *args, **kwargs):
        input_value, y_target = batch
        y_pred = self.do_validation_inference(input_value)
        logger.debug(f"{y_pred} and {y_target}")
        loss = self.loss(y_pred, y_target)  # Compute the loss
        y_pred = self.postprocess_eval(y_pred)
        if self.logger:
            for metric_name, metric in self.metrics_dict_val.items():
                self.log(
                    f"Validation/{metric_name}_step",
                    metric(
                        torch.argmax(y_pred, dim=1),
                        y_target,
                    ),
                )
            # We log to progress bar and pass it on
            self.log("val_loss", loss.detach(), prog_bar=True)
        return {"val_loss": loss}

    def do_validation_inference(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(x)

    def test_step(self, batch, *args, **kwargs):
        input_value, y_target = batch
        y_pred = self.do_validation_inference(input_value)
        logger.debug(f"{y_pred} and {y_target}")
        loss = self.loss(y_pred, y_target)  # Compute the loss
        y_pred = self.postprocess_eval(y_pred)
        if self.logger:
            for metric_name, metric in self.metrics_dict_val.items():
                self.log(
                    f"Test/{metric_name}_step",
                    metric(
                        torch.argmax(y_pred, dim=1),
                        y_target,
                    ),
                )
            # We log to progress bar and pass it on
            self.log("test_loss", loss, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self, **kwargs):
        """Define the optimizer and learning rate scheduler"""
        self.optimizers = [
            AdamW(self.parameters(), lr=self.lr),
        ]
        self.schedulers = []
        return self.optimizers, self.schedulers

    def training_epoch_end(self, epoch_output):
        for metric_name, metric in self.metrics_dict_train.items():
            self.log(f"Training/{metric_name}_epoch", metric.compute())

    def validation_epoch_end(self, outputs):
        for metric_name, metric in self.metrics_dict_val.items():
            self.log(f"Validation/{metric_name}_total", metric.compute())

    def test_epoch_end(self, outputs):
        for metric_name, metric in self.metrics_dict_val.items():
            self.log(f"Test/{metric_name}_total", metric.compute())
        return outputs
