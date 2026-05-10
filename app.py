import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Set current directory as base
base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=base_dir)
CORS(app)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

current_df = None

# Serving index.html safely
@app.route('/')
def index():
    return send_from_directory(base_dir, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        if file.filename.endswith('.csv'):
            current_df = pd.read_csv(file)
        else:
            current_df = pd.read_excel(file)
        
        return jsonify({
            "message": "File uploaded successfully!",
            "columns": list(current_df.columns),
            "rows": len(current_df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global current_df
    user_query = request.json.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        if current_df is not None:
            prompt = f"Data Context:\n{current_df.head(10).to_string()}\n\nQuestion: {user_query}"
        else:
            prompt = f"General Question: {user_query}"

        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
