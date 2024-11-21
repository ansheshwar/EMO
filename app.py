import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, request, jsonify

# Load environment variables from .env file if it exists
load_dotenv()

# Load data function
def load_data(file_path):
    """Load the Emotion dataset."""
    df = pd.read_csv(file_path)
    return df

# Emotion Classifier class
class EmotionClassifier:
    def __init__(self):
        self.model = None
    
    def train(self, data):
        """Train the emotion detection model."""
        X = data['content'] 
        y = data['sentiment']  
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())
        self.model.fit(X_train, y_train)

    def predict(self, text):
        """Predict emotion from input text."""
        return self.model.predict([text])[0]

# Emotion Analyzer class
class EmotionAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')  # Ensure you set this environment variable
        self.llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7)
        self.classifier = EmotionClassifier()

    def analyze_emotion(self, text):
        """Analyze emotion from text using Google API and classifier."""
        
        # First use classifier to predict emotion
        predicted_emotion = self.classifier.predict(text)
        
        # Use LangChain with Gemini to get suggestions based on predicted emotion
        prompt = (f"Suggest an action for someone feeling {predicted_emotion}. "
                  f"Also analyze this suggestion and provide a unique response.")
        
        response = self.llm.invoke(prompt)
        
        return predicted_emotion, response.content.strip()

# Initialize Flask app
app = Flask(__name__)

# Load dataset and train classifier when starting the app
df = load_data('data.csv')  # Ensure this path is correct
analyzer = EmotionAnalyzer()
analyzer.classifier.train(df)

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze text for emotions."""
    data = request.json
    
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    user_input = data['text']
    
    predicted_emotion, suggestion = analyzer.analyze_emotion(user_input)
    
    return jsonify({
        'predicted_emotion': predicted_emotion,
        'suggestion': suggestion
    })

if __name__ == "__main__":
    app.run(debug=True)