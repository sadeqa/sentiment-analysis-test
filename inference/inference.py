""" File to infer on test set"""
import pandas as pd
import fire
from loguru import logger
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from experiment.sentiment_model import SentimentModel
from experiment.sentiment_tokenizer import SentimentTokenizer
from experiment.sentiment_datamodule import LABEL_TO_ID


class SentimentInference:
    def __init__(self,
                 pretrained_model_name_or_path: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                 max_length: int = 128,
                 ):
        logger.info("Initializing Tokenizer and Model")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = SentimentModel(pretrained_model_name_or_path=pretrained_model_name_or_path)
        logger.info("Done.")
        self.id_to_label = {v:k for k,v in LABEL_TO_ID.items()}
        self.max_length = max_length

    def setup_infer(self, checkpoint_path: str, device: str):
        """ setup model for inference"""
        logger.info(f"Loading model from {checkpoint_path}")
        self.model.load_model(checkpoint_path)
        logger.info("Done")
        self.model.to(device)
        self.model.eval()

    def infer_csv(self, file_path: str, checkpoint_path: str, device: str):
        """ infer on a csv file"""
        df = pd.read_csv(file_path)
        self.setup_infer(checkpoint_path, device)
        predictions = []
        for row in tqdm(df.values):
            input_tensor = self.tokenizer(
                row[0],
                return_tensors='pt',
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            input_tensor = {k: v.to(device) for k, v in input_tensor.items()}
            output = self.model.forward(input_tensor).detach().cpu().numpy()
            predictions.append(self.id_to_label[np.argmax(output)])
        output_df = pd.DataFrame(predictions, columns=["prediction"])
        output_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    cli = SentimentInference()
    fire.Fire(cli)