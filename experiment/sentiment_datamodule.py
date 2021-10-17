""" File to define Dataset and Dataloader"""
import os
from typing import Optional, Tuple, List, Dict

import pandas as pd
import torch
from loguru import logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from experiment.sentiment_tokenizer import SentimentTokenizer

DEFAULT_TRAIN_DATA_PATH: str = os.path.join(
    "data", "train_clean.csv"
)

DEFAULT_VAL_DATA_PATH: str = os.path.join("data", "val_clean.csv")
LABEL_TO_ID: Dict = {"negative": 0, "neutral": 1, "positive": 2}


class SentimentsDataset(Dataset):
    def __init__(
            self,
            processed_data_path: str,
            num_classes: int = 3,
            preprocessors=None,
    ) -> None:
        """initialize the dataset class
        """
        super().__init__()
        if preprocessors is None:
            preprocessors = []
        if processed_data_path is not None and os.path.exists(
                processed_data_path
        ):
            raw_data = pd.read_csv(processed_data_path)
            self.data: List = [
                [
                    item[1].content,
                    item[1].sentiment,
                ]
                for i, item in tqdm(enumerate(raw_data.iterrows()))
            ]
        self.num_classes = num_classes
        self.preprocessors = preprocessors

    def __len__(self) -> int:
        """Get the length of the dataset

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Dict, torch.Tensor]:
        """Get input for sentiment model

        Args:
            index (int): The index of the document to get

        Returns:
            Tuple[Dict, torch.Tensor]: The input for the model and the label
        """
        doc = self.data[index]
        label: torch.Tensor = self.get_label(doc[1])
        for preprocessor in self.preprocessors:
            doc = preprocessor(doc)
        return doc, label

    def get_label(self, doc) -> torch.Tensor:
        return torch.Tensor([LABEL_TO_ID[doc]]).long().squeeze(0)


class SentimentDatamodule(LightningDataModule):
    def __init__(
            self,
            train_data_path: str = DEFAULT_TRAIN_DATA_PATH,
            val_data_path: str = DEFAULT_VAL_DATA_PATH,
            batch_size: int = 2,
            num_workers: int = 0,
            model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            max_length: int = 256,
            **kwargs,
    ):
        """
        init of Datamodule
        """
        super().__init__(**kwargs)
        logger.info("Initializing Preprocessors")
        self.transforms = [
            SentimentTokenizer(model_name=model_name, max_length=max_length)
        ]
        logger.info("Done.")

        self.train_data_path: str = train_data_path
        self.val_data_path: str = val_data_path

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.sampler = None

        self.train_dataset: Optional[SentimentsDataset] = None
        self.val_dataset: Optional[SentimentsDataset] = None
        self.test_dataset: Optional[SentimentsDataset] = None

    def setup(self, stage: Optional[str] = None):
        """Step where we setup the datasets
        => by that time, every needed input has been incorporated

        Args:
            stage (str, optional): [description]. Defaults to None.
        """
        logger.info(f"setup for stage: {stage}")

        if stage == "fit" or stage is None:
            self.train_dataset = SentimentsDataset(
                processed_data_path=self.train_data_path,
                preprocessors=self.transforms,
            )
            self.val_dataset = SentimentsDataset(
                processed_data_path=self.val_data_path,
                preprocessors=self.transforms,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SentimentsDataset(
                processed_data_path=self.val_data_path,
                preprocessors=self.transforms,
            )

    def train_dataloader(self):
        if self.sampler is None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=self.sampler,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
