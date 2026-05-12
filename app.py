import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

# Import the class we created in Data_processor.py
from data_processor import DataProcessor

# Initialize Flask (template_folder='.' looks for index.html in the main folder)
app = Flask(__name__, template_folder='templates')
CORS(app)

df = None

@app.route('/')
def home():
    """Route to serve the frontend."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"File index.html not found: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Route to handle CSV upload and cleaning via DataProcessor."""
    global df
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        # 1. Read raw CSV
        raw_df = pd.read_csv(file)
        
        # 2. Use our DataProcessor class to clean data
        processor = DataProcessor(raw_df)
        df = processor.clean_data()
        stats = processor.get_info()
        
        return jsonify({
            "message": "Success: Cleaned by DataProcessor Engine!",
            "rows": stats['rows'],
            "cols": stats['cols']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Route for the AI Agent to explain the data."""
    global df
    data = request.json
    query = data.get('query', '')

    if df is None:
        return jsonify({"response": "Error: Please upload a file first."})

    # Prepare context for AI
    context = f"The dataset has {len(df)} rows and {len(df.columns)} columns. All missing values are cleaned."

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"Context: {context}. User Question: {query}. Task: Provide a brief analysis."
            response = model.generate_content(prompt)
            return jsonify({"response": response.text})
        
        return jsonify({"response": f"Local Discovery: {context}"})
    except Exception as e:
        return jsonify({"response": f"Automated Insight: {context}"})

if __name__ == '__main__':
    # Get port from environment for deployment, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
