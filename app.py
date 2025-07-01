from flask import Flask, request, jsonify, render_template
import joblib
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

app = Flask(__name__) # initialize the flask API

# Configure logging to store info about each step
handler = RotatingFileHandler('spam_detection.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Load the model and the vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


@app.route('/')
def index():
    return render_template('index.html') #read the html file


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400 # check if message key is present else indicate in the log file no message provided

        message = data['message']
        message_counts = vectorizer.transform([message]) #vectorize the message user provided from text to numeric
        message_counts_dense = message_counts.toarray()  # Convert sparse matrix to dense array
        prediction = model.predict(message_counts_dense) #make prediction

        app.logger.info(f'Message: {message} | Prediction: {prediction[0]}') #add info on whatever we have done to the log file

        # Convert numpy int64 to Python int
        prediction_int = int(prediction[0])

        return jsonify({'prediction': prediction_int}) #if an error happens during prediction add info on that to the log file
    except Exception as e:
        app.logger.error(f'Error: {str(e)}')
        return jsonify({'error': 'An error occurred during prediction'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

# Runs the Flask app if the script is executed directly.
# Enables debug mode, which provides detailed error messages and auto-reloads the server on code changes.
if __name__ == '__main__':
    app.run(debug=True)
