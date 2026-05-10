import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# AI Configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Global variable to store the dataframe
current_df = None

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
            # Case 1: Data Analysis & Cleaning (When file exists)
            prompt = f"""
            You are a Data Expert. Analyze the following data snippet (first 10 rows):
            {current_df.head(10).to_string()}
            
            User Question: {user_query}
            
            Task: If the user asks for data cleaning, error detection, or analysis, provide a detailed technical response. 
            If they ask a general question, use the data context if relevant.
            """
        else:
            # Case 2: General Chat (When NO file is uploaded)
            prompt = f"""
            The user is asking a general question: {user_query}
            Please provide a helpful and professional response as a technical assistant.
            """

        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
