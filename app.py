import os
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Securely load API Key from Render Environment Variables
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("System Error: GOOGLE_API_KEY is not configured")

# Initialize Gemini 1.5 Flash Model
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        
        # Check all common field names to ensure we get the message
        user_message = data.get('message') or data.get('prompt') or data.get('text')
        
        if not user_message:
            return jsonify({'error': 'No input detected. Please type a message.'}), 400

        # Generate response from AI
        response = model.generate_content(user_message)
        
        if response and response.text:
            return jsonify({'response': response.text})
        else:
            return jsonify({'error': 'AI could not generate a response.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Port 10000 is required for Render deployments
    app.run(host='0.0.0.0', port=10000)
