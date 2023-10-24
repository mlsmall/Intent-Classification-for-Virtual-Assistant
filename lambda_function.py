import awsgi
from flask import Flask, request, jsonify

import boto3
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load the pre-trained model from S3
s3 = boto3.client('s3')
bucket = 'mygvabucket'
model_key = 'bert_model.pt'

s3.download_file(bucket, model_key, 'bert_model.pt')

model = torch.load('bert_model.pt')

def classify(text):
    encoded_inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**encoded_inputs)
    logits = outputs.logits
    predictions = logits.argmax(dim=1)

    prediction = predictions.item()

    return prediction

@app.route("/classify", methods=["POST"])
def classify_api():
    text = request.json["text"]

    prediction = classify(text)

    response = {
        "prediction": prediction,
    }

    return jsonify(response)

def lambda_handler(event, context):
    app.wsgi_app = awsgi.WSGIApplication(app)
    return app.wsgi_app(event, context)