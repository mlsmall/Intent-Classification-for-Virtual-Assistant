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
To test the classifier run the following
```pip install transformers```
and install PyTorch (https://pytorch.org/get-started/locally/)

Then create a python file and run the following:
```
text = "I need help setting up an account" # Type an utterance that would correspond to one of the ten intents listed above.

import transformers
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="maurosm/bert_classification_model")
classifier(text)
```

The first time your run your file, a pytorch_model.bin will be downloaded. This file contains the parameters and weights of the trained model. 
To keep testing with different utterances just change the ```text``` in the code and run the python file again.

