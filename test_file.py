text = "I need help with registration problems" # Type in your text

import transformers
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="maurosm/bert_classification_model")
print(classifier(text))