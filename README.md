# Sentiment Analysis Test

- [Sentiment Analysis Test](#sentiment-analysis-test)
  - [🎯 Goals](#user-content--goals)
  - [📊 Datasets](#user-content--datasets)
  - [📖 Rules](#user-content--rules)
  - [👩‍💻 What to do](#user-content--what-to-do)

## 🎯 Goals

Train a multilingual text classifier that predicts the sentiment polarity of a given text.

Possible sentiments:
* `positive`
* `negative`
* `neutral`

## 📊 Datasets

* `data/train.csv`: a training dataset containing 25k multilingual texts annotated with their corresponding sentiment
* `data/test.csv`: a test dataset containing 2500 multilingual texts

## 📖 Rules

* Code should be written in Python 3
* Code should be easily runnable, provide a pip requirements.txt file or a conda environment.yml file describing code dependencies
* Code should be documented to explain your methodology, choices, and how to run and get results
* Code should output a file `predictions.csv`, containing the predictions of your best classifier on the test dataset

## 👩‍💻 What to do

1. Fork the project via `github`
2. Clone your forked repository project https://github.com/YOUR_USERNAME/sentiment-analysis-test
3. Commit and push your different modifications
4. Send us a link to your fork once you're done!

## 🤖 Installation
To setup your project:

you can create your environment and install all the requirements using pip:
```
pip install -r requirements/requirements.txt
```

## 📖 Approach
For all intermediary studies, you can refer
to the notebooks folder with all the analysis work done there.

For the data analysis, you can refer to [data_analysis](https://github.com/sadeqa/sentiment-analysis-test/blob/master/notebooks/data_analysis.ipynb).

We can see from the data analysis that our datasets are multilingual 
with more than 50 different languages. The majority of the sentences are
around 18 words in size meaning that an approach with transformers can work pretty 
well. (Large documents are hard to handle using transformers in a simple way).
Also looking at the sentences, we can see that they extracted from a social network.

The first idea is to use a Bert like model such as XLM-Roberta that handle
multilingual data and finetune it on our dataset. Due to some lack of
computing power and the fact that the sentences look like tweets, we will 
use a pretrained XLM-Roberta on sentiment detection on tweets [model page](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) 
and simply freeze all the backbone layers as they are already trained to
extract the important signal and only finetune the classifier's weights.

### Run the training
To run the training, you can change the parameters you want in the 
[sentiment_config.yaml](https://github.com/sadeqa/sentiment-analysis-test/blob/master/config/sentiment_config.yaml) 
and run the following:
```
# for training with params in config
make train

# for training on cpu without changing the config 
make train_cpu

# for training on gpu without changing the config 
make train_gpu
```

For the results on the training you can refer to the notebooks folder 
with an evaluation of both pretrained and finetuned models. We evaluate 
the model with respect to each class but also with respect to each language.

#### Results of finetuned model on 5 most present languages on validation set (20% of train)
| lang  | accuracy  | size |   
|---|---|---|
|  English | 0.8147 |  831 |   
|  Russian | 0.9422 |  657 |
|  Indonesian | 0.8665 |  629 |
|  Arabic | 0.8502 |  347 |
|  French | 0.8510 |  302 |

### Improvement proposition:
One possible way to improve the model performance is by augmenting the data either
by labeling new data or generating automatically new ones. One of the most 
common way of doing so is by translation. Normally, translating the more represented
languages to the less represented ones to help the model perform better on those.
In here, we propose to generate augmented sentences 
for the english language as it lacking compared to the others 
by translating French sentences into english ( Translation model has good performance)
and keeping the same label as the meaning remains the same. 

#### Results of finetuned augmented model on 5 most present languages on validation set (20% of train)
| lang  | accuracy  | size |   
|---|---|---|
|  English | 0.8148 |  831 |   
|  Russian | 0.9376 |  657 |
|  Indonesian | 0.8648 |  629 |
|  Arabic | 0.8357 |  347 |
|  French | 0.8481 |  302 |



##  Models
The best model finetuned on train, can be downloaded [here](https://drive.google.com/file/d/1ilSiREEcshWA49ks0_57cOVA8RO8b4Cc/view?usp=sharing).

The best model finetuned on train augmented can be downloaded [here](https://drive.google.com/file/d/1BRqRcVmFqe1zaqJWsUyFpl6eK3fWD6Xq/view?usp=sharing).

## Run Inference
To run inference, you can run:
```
make inference
```
Just make sure the paths for data and models are correct in the Makefile and 
the device to run on.
