import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configuration: Setting the static folder to current directory
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# AI Setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

current_df = None

# Route to serve the index.html file (Fixes 404 error)
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

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
            # Context for data analysis
            prompt = f"Analyze this data snippet:\n{current_df.head(10).to_string()}\n\nUser Question: {user_query}"
        else:
            # Context for general chat
            prompt = f"User Question: {user_query}\nAnswer as a professional technical assistant."

        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
