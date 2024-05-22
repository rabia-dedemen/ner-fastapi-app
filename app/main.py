from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import pickle
import joblib
import function_util
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


app = FastAPI()

model_folder = r"C:\Users\ASUS\PycharmProjects\ner-fastapi\app\model\modelfolder\models"

tokenizer = AutoTokenizer.from_pretrained(model_folder)
with open(r"C:\Users\ASUS\PycharmProjects\ner-fastapi\app\model\dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

with open(r"C:\Users\ASUS\PycharmProjects\ner-fastapi\app\model\model.pkl", "rb") as f:
    aspect_model = pickle.load(f)
sentiment_model = joblib.load(r'C:\Users\ASUS\PycharmProjects\ner-fastapi\app\model\model2.joblib')


class ReviewRequest(BaseModel):
    review: str


class Aspect(BaseModel):
    aspect: str
    comments: List[str]


class Sentiment(BaseModel):
    sentiment: int


def predict_sentiment(input_string):
    # Tokenize the input string
    input_tokens = tokenizer(input_string, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = sentiment_model(**input_tokens)

    predicted_class = torch.argmax(outputs.logits).item()

    return predicted_class


@app.get('/')
def read_root():
    return {'message': 'ABSA Model API'}


@app.post("/predict_aspect/")
async def predict_aspect(review_request: ReviewRequest):
    review_text = review_request.review

    processed_review = function_util.preprocess(review_text)
    bow_review = dictionary.doc2bow(processed_review)

    topics = aspect_model.get_document_topics(bow_review)

    main_topic = max(topics, key=lambda x: x[1])[0]
    topic_name = function_util.get_topic_name(main_topic)

    return {"review": review_text, "predicted_aspect": topic_name}


@app.post("/analyze_sentiment/")
async def analyze_sentiment(comments: List[str]):
    sentiments = []

    for comment in comments:
        sentiment_prediction = predict_sentiment([comment])
        sentiments.append(Sentiment(sentiment=sentiment_prediction))

    return sentiments


# fix reqs
