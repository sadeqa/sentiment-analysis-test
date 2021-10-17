""" Tokenizer for sentiment detection"""
from typing import Dict, List

from loguru import logger
from transformers import AutoTokenizer
import torch


class SentimentTokenizer:
    def __init__(self,
                 model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                 max_length: int = 256,
                 ):
        """ init function"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length: int = max_length

    def __call__(self, text_input: List[str]) -> Dict[str, torch.tensor]:
        """ tokenizer string input into torch tensors"""
        logger.debug(f"Text Input: {text_input[0]}. Label is: {text_input[1]}")
        tokenized = self.tokenizer(
            text_input[0],
            return_tensors='pt',
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        return tokenized
