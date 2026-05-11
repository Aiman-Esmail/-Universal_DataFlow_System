import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Global variable to hold data in memory
df = None

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route('/')
def home():
    """Renders the main dashboard."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles CSV upload and initial data discovery."""
    global df
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Load data
        df = pd.read_csv(file)
        
        # Initial Automated Cleaning
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        cleaned_rows = initial_rows - len(df)
        
        return jsonify({
            "message": "Success: File Loaded!",
            "rows": len(df),
            "cols": len(df.columns),
            "duplicates_removed": cleaned_rows
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Agent logic for cleaning and AI-powered insights."""
    global df
    data = request.json
    user_query = data.get('query', '')

    if df is None:
        return jsonify({"response": "Error: Please upload a dataset first."})

    # Internal Data Audit
    null_counts = df.isnull().sum().sum()
    cols = list(df.columns)
    
    # Simple Clean Action: Forward fill missing values
    df.fillna(method='ffill', inplace=True)

    # Prepare Context for the AI Agent
    context = (
        f"The user uploaded a dataset with {len(df)} rows and {len(cols)} columns. "
        f"Columns identified: {', '.join(cols)}. "
        f"Agent performed: Removed duplicates and filled {null_counts} missing values."
    )

    # Use Gemini if API KEY exists
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            full_prompt = f"Context: {context}\nUser Question: {user_query}\nTask: Explain the data structure and the cleaning steps taken."
            response = model.generate_content(full_prompt)
            return jsonify({"response": response.text})
        else:
            return jsonify({"response": f"System Report: {context}"})
    except Exception as e:
        return jsonify({"response": f"Automated Summary: {context}"})

# ---------------------------------------------------------
# SERVER STARTUP
# ---------------------------------------------------------

if __name__ == '__main__':
    # Important: Render provides the PORT as an environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
