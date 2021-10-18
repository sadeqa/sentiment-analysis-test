SHELL := /bin/bash
.PHONY : all

help:
	cat Makefile

train:
	python -m experiment.sentiment_cli --config config/sentiment_config.yaml

train_gpu:
	python -m experiment.sentiment_cli --config config/sentiment_config.yaml --trainer.gpus 1 --trainer.auto_select_gpus true

train_cpu:
	python -m experiment.sentiment_cli --config config/sentiment_config.yaml --trainer.gpus null --trainer.auto_select_gpus false

inference:
	python -m inference.inference infer_csv data/test.csv models/model-epoch\=9-val_loss\=0.40.ckpt cuda

