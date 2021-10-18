""" File to augment english data with translation from french"""
import pandas as pd
import fire
from langdetect import detect, LangDetectException
from loguru import logger
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


def detect_language(s):
    try:
        return detect(s)
    except LangDetectException:
        return "unknown"


class French2EnglishAugmentation:
    def __init__(self):
        logger.info('Loading tokenizer and model ...')
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        logger.info('Done.')

    def translate(self, sentence: str):
        """ translate french sentence to english"""
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation="longest_first")
        output = self.model.generate(**inputs)[0]
        english_sentence = self.tokenizer.decode(output, skip_special_tokens=True)
        return english_sentence

    def augment_csv(self, csv_path: str, output_file_path: str):
        """ Augment csv with english data translated from french"""
        df = pd.read_csv(csv_path)
        augmented_data = []
        for row in tqdm(df.values):
            lang = detect_language(row[0])
            if lang == "fr":
                new_sent = self.translate(row[0])
                augmented_data.append([new_sent, row[1]])

        logger.info(f"{len(augmented_data)} data point was generated.")
        augment_df = pd.DataFrame(augmented_data, columns=df.columns)
        final_df = df.append(augment_df)
        logger.info(final_df.head())
        final_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    cli = French2EnglishAugmentation()
    fire.Fire(cli)
