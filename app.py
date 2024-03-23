from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Load the trained model
model = joblib.load("model/naive_bayes.pkl")


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Get the review input from the form
        review = request.form['review']
        # Clean the user input
        clean_input = preprocess_text(review)
        # Perform sentiment analysis on the cleaned input
        prediction = model.predict([clean_input])
        # Map the prediction to sentiment labels
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        # Render the result template with the predicted sentiment
        return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)
