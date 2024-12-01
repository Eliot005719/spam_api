from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Step 1: Train the model
data = pd.DataFrame({
    'Label': ['spam', 'ham', 'spam', 'ham', 'ham', 'spam'],
    'Message': [
        'Win a $1000 now!', 
        'Hey, how are you?', 
        'Congratulations, you won!', 
        'Meeting at 5 pm.', 
        'Looking forward to our vacation!', 
        'Get a free gift card today!'
    ]
})
data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})  # Encode labels
X = data['Message']
y = data['Label']

# Vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X_transformed = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_transformed, y)

# Step 2: Define the Flask endpoint
@app.route('/classify', methods=['POST'])
def classify_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Transform and classify
    input_transformed = vectorizer.transform([message])
    prediction = model.predict(input_transformed)[0]
    probabilities = model.predict_proba(input_transformed)[0]
    spam_probability = probabilities[1] * 100  # Spam probability in percentage
    
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    return jsonify({
        'message': message,
        'classification': label,
        'spam_probability': spam_probability
    })

# Step 3: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
