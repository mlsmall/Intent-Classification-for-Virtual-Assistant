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
To test the classifier:
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

### Use an API
The option to use an API was attempted but based on size limitations of AWSLambda's free tier, the dependencies could not be uploaded.  To create a simple API you can use a server like AWSLambda and a framework like Flask.  For this model, the required lambda function is [here](https://github.com/mlsmall/Text-Classification-for-Virtual-Assistant/blob/main/lambda_function.py). The model weights and parameters that were saved from the model were uploaded to an Amazon S3 server as a PyTorch file. This file can be [download here](https://mygvabucket.s3.ca-central-1.amazonaws.com/bert_model.pt?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjECMaCXVzLWVhc3QtMSJIMEYCIQDqT2%2B2v6TAloW5NAdJ2GCRdsg8jn6c47AvhovN2vyLQgIhALVhDMnFX0bC58mkbFL%2Fy57TNP6%2FMn6Lted6UINeu8VIKuQCCEwQBBoMMjc3MTcyNDE4MzM5Igz%2FQvTUJoBcKbeJUq4qwQIuWdc6rsUAH8l2gokPLt%2BaRBE3FTV4s4n4Bse27TnOI2%2FCefYveFFTPD4vmJ1DIxCEuMNfP4uL%2BZOxHFGBwG9ByLId6EjmmDZv2Gqqga%2FHKKg0V9IoImp%2Ffh0EYqHxMy1MQZ%2FVXHPVZzkkeNMD8%2BBuSiKl%2FyU0%2BvyGSYcTPLdudqGPVeVlvIo%2FJ66VZhQ%2BYiECXxppPA6u6JqdcmxgxalRxReWHpppqCPtTZ%2F5o%2FHvSJDkedvid1wBsRBKW%2FSH9aBJ5GwPJB4sJPxSkQWR2RdXARmXMU%2FwpqEISqRf12RpKa%2BdU92DNGRoWOOubuLev%2BVTVh3cYOp%2BjpSJUZRD3UM79IgOKSfQTCwj0iGgdEFVRX1auJtW%2FtscjMhEXLoo51QWSyDLK8USmxkuIeb5mt8%2FBuIuaLcLCnSQY2WN0rDFmg4wmaPgqQY6sgI51tASWgg%2B5tTScp3WGCPLsWsk4inPRQPCLySF%2BwW96slGQGqug%2BT2oANuitgUSy2qKC3Uk4QBp0j%2Fwse12dE%2FgKZ4qz6UULdFT%2B2rA1rMcGWDQT6a5uKTl1GOPBJtdPMrYhwbL74R8fLScuPSxrADDA417Nr%2Ba9VkD9KFWIR1rW7rEaahG6p5raeohUz3OBvts9Q8tc14oUIufkAr%2B6FnWJo2cP2n8VVQpi7PTvWDnc9p1WrGDwv8jVfaX4bQLO%2BTHXaRv5WdUnqBRbPwa5%2BpUBf6SIdLTkuvGqrzbKkiDysh5B0XOHiZ0hwLNSY%2BTGns9aOYwiXUMY6c9gnPjOvZnD2Iful%2BLYoJXVsc%2BbogXux6WBJ8ncHj1GsvOfqcHSye9uFYFJ9iwHVtF%2BcZfiASGWU%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231024T190738Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUBCGDO4RW6SISEPX%2F20231024%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Signature=42b251322ebb2b79a493810dc47c776370c1c11cac0216bfad4b4ba7a0eeab71).
To create an API, the lambda function has to be uploaded with the dependencies or packages listed in the function. To get the required dependencies run the following commands in a terminal:
```
mkdir dependencies
pip3 install --target=dependencies transformers torch boto3 aws-wsgi flask
```
Then zip the newly created dependencies folder, add the lambda_function.py file to the zip file, and upload them to AWSLambda.

## Evaluation
### Model Performance Overview
In this section, we assess the performance of our text classifier in determining the user intents. The evaluation focuses on the model's challenges when dealing with intents that are closely related. It's important to acknowledge that while the model shows promising results in many cases, the results are somewhat underwhelming  when handling similar intents.

### Challenges with Similar Intents

One of the challenges encountered is the classification of intents that are closely related. When intents share common keywords or context, the model can become confused. This is a common problem in natural language understanding. Following are examples where the model struggled and misclassified the intent.

#### Example 1: 
* Utterance: I need help with registration.
* Predicted Intent: *create_account*
* Expected Intent: *registration_problems*

The model's misclassification in this case could be the presence of the word "registration" in the utterance. Both the predicted intent "create_account" and the expected intent "registration_problems" are associated with the term "registration”.  It’s important to note that the utterance “I need help with registration”, could also correspond to the intent “create_account”, since the customer may say this because he or she is trying to create a new account.

#### Example 2: 
* Utterance: i want to change my existing account
* Predicted Intent: *delete_account*
* Expected Intent: *edit_account*

The user's input, "I want to change my existing account," is somewhat ambiguous in terms of the specific action they intend to take. While the user does express a desire to modify their account, the model may have interpreted "change" as synonymous with "delete" in this context. This ambiguity can lead to the model's choice of "delete_account" as the predicted intent.

#### Example3: 
* Utterance: i would like my money back
* Predicted Intent: *get_invoice*
* Expected Intent: *get_refund*

The user's statement, "I would like my money back," is a common expression used to request a refund. However, the model may have relied on keywords associating "money" with financial transactions like invoices, rather than recognizing it as a refund request.

### Improving the Model
It's essential to recognize that the accuracy of this model was greatly dependent on the quality of the dataset. The dataset was sourced from Kaggle without augmenting or improving it. This limited the model's ability to generalize effectively to real-world scenarios and handle cases with nuances.

To improve this model’s performance, it is recommended to provide this model with a more diverse dataset that includes a wide range of examples for each intent. This would specifically mean adding more elaborate and detailed examples to the training data. This would help the model learn how to distinguish between similar intents that are frequently confused.

## Model description

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on a dataset provided by Kaggle.
DistilBERT is a smaller and efficient variant of the BERT model. It was developed by Hugging Face and retains much of BERT's language understanding capabilities while significantly reducing its size. It was pre-trained on a vast amount of text data and excels at various natural language processing tasks, making it a great choice for applications where memory and processing resources are limited.
  
It was trained on a Google Collab jupyter notebook using a T4 GPU. The notebook can be found [here](https://github.com/mlsmall/Text-Classification-for-Virtual-Assistant/blob/main/text_classification_model.ipynb)

## Training and evaluation data
There were traininig samples 13,155 samples, of which 10,524 were used for training and 2,631 were used for validation. It was an 80-20 split.

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
