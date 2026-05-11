import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# Global variable to store the dataframe
df = None

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        # Immediate basic cleaning
        df.drop_duplicates(inplace=True)
        return jsonify({"message": "Success: File Loaded!", "rows": len(df), "cols": len(df.columns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    global df
    data = request.json
    user_query = data.get('query', '')

    if df is None:
        return jsonify({"response": "Error: No data found. Please upload a file first."})

    # Data Discovery Logic
    null_counts = df.isnull().sum().sum()
    column_names = ", ".join(df.columns.tolist())
    
    # Simple automated cleaning for the demo
    df.fillna(method='ffill', inplace=True)

    analysis_report = (
        f"Data Insights: Identified {len(df)} rows and {len(df.columns)} columns. "
        f"Columns found: {column_names}. "
        f"Automatic cleaning: Fixed {null_counts} missing values using forward-fill."
    )

    # Integrating with Gemini for the "Smart Agent" feel
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"User Query: {user_query}. Context: {analysis_report}. Explain what was done to the data."
            response = model.generate_content(prompt)
            return jsonify({"response": response.text})
        else:
            return jsonify({"response": f"Agent Action: {analysis_report}"})
    except Exception as e:
        return jsonify({"response": f"Local Analysis: {analysis_report}"})

if __name__ == '__main__':
    app.run(debug=True)
