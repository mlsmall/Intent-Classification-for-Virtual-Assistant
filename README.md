# Multi-Class-Text-Classification-for-Virtual-Assistant

This project consits of training a text classifier for a virtual assistant using a dataset that consists of 13,000 utterances and their corresponding intents. The dataset was acquired from Kaggle and can be found [here](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants). This classifier predicts an intent based on an utterance given by the user.  This particular model was limited to only 10 intents.

The 10 classes or intents are:
* change_shipping_address
* complaint
* contact_customer_service
* create_account
* delete_account
* edit_account
* get_invoice
* get_refund
* payment_issue
* registration_problems

## Testing the Classifier
#### 1) Using the HuggingFace Rest API
Go to [this link](https://huggingface.co/spaces/maurosm/intent-class) and type an utterance that corresponds to one of the intents listed above.

#### 2) Testing it locally on your machine
You also have the option of running an inference object locally on your machine.
* From a terminal run ```pip install transformers```
* Install PyTorch (https://pytorch.org/get-started/locally/)
* Create a python file and run the following:
```
text = "I need help setting up an account" # Type an utterance that would correspond to one of the ten intents listed above.

import transformers
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="maurosm/bert_classification_model")
classifier(text)
```
The first time the ```classifier``` line above is run, a pytorch_model.bin file will be downloaded. This file contains the parameters and weights of the trained model. 
To keep testing with different utterances just change the ```text``` in the code and run the python file again.

## Model description

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on a dataset provided by Kaggle.
DistilBERT is a smaller and efficient variant of the BERT model. It was developed by Hugging Face and retains much of BERT's language understanding capabilities while significantly reducing its size. It was pre-trained on a vast amount of text data and excels at various natural language processing tasks, making it a great choice for applications where memory and processing resources are limited.

Distilbert was used because it's favorable for sentiment analysis tasks, and its smaller size makes it more practical for faster inference, resource-constrained applications and real-time NLP tasks. You can read the [paper for Distilbert](https://arxiv.org/abs/1910.01108) to learn more.
  
The model was trained on a Google Collab jupyter notebook using a T4 GPU. The notebook can be found [here](https://github.com/mlsmall/Text-Classification-for-Virtual-Assistant/blob/main/text_classification_model.ipynb)

## Training and evaluation data
The dataset had 11,693 training samples, of which 9,354 were used for training and 2,339 were used for validation for a 80-20 split.

### Training hyperparameters
The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 16
- eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- num_epochs: 2

### Training results
The model achieves the following results against the validation set:
- Loss: 0.0053
- Accuracy: 0.9992

| Training Loss | Epoch | Validation Loss | Accuracy |
|:-------------:|:-----:|:---------------:|:--------:|
| 0.217         | 1.0   | 0.0081          | 0.9989   |
| 0.0059        | 2.0   | 0.0053          | 0.9992   |
