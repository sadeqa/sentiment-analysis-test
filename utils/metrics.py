""" File with metrics defined"""
import torch
from loguru import logger
from torchmetrics.classification import F1, Accuracy, Precision, Recall


def classification_metrics(n_classes: int = 2):
    """Function to set up the classification metrics"""
    logger.info(f"Setting up metrics for: {n_classes}")
    metrics_dict_train = torch.nn.ModuleDict(
        {
            "accuracy": Accuracy(),
            "recall": Recall(),
            "precision": Precision(),
            "F1": F1(),
        }
    )
    metrics_dict_val = torch.nn.ModuleDict(
        {
            "accuracy": Accuracy(),
            "recall": Recall(),
            "precision": Precision(),
            "F1": F1(),
        }
    )
    return metrics_dict_train, metrics_dict_val
