# main.py
#pip install nltk
#pip install fastapi uvicorn
#uvicorn main:app --reload
from fastapi import FastAPI
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

app = FastAPI()

# Load the trained model
with open('Extra_Tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your fitted vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back to form text
    text = ' '.join(filtered_tokens)

    # Remove extra spaces
    text = re.sub(' +', ' ', text)

    return text

class Text(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def predict_sentiment(text: Text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text.text)
    
    # Transforming the text data to TF-IDF vector
    tfidf_text = vectorizer.transform([preprocessed_text])
    
    # Make prediction
    prediction = model.predict(tfidf_text)
    
    # Convert numeric label to string label
    sentiment = labels[prediction[0]]
    
    return {"prediction": sentiment}
