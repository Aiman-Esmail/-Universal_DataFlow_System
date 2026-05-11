import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

# Important: template_folder='.' because index.html is in the root folder
app = Flask(__name__, template_folder='.')
CORS(app)

df = None

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error: index.html not found. Details: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file)
        # Auto-cleaning duplicates
        df.drop_duplicates(inplace=True)
        return jsonify({
            "message": "Success: File Loaded!", 
            "rows": len(df), 
            "cols": len(df.columns)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    global df
    data = request.json
    query = data.get('query', '')
    
    if df is None:
        return jsonify({"response": "Error: Upload data first."})
    
    # Logic for Data Discovery
    null_count = df.isnull().sum().sum()
    df.fillna(method='ffill', inplace=True)
    
    context = f"Dataset: {len(df)} rows. Agent cleaned {null_count} missing values."
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"Context: {context}. User: {query}. Brief technical summary:"
            response = model.generate_content(prompt)
            return jsonify({"response": response.text})
        return jsonify({"response": f"Local Report: {context}"})
    except:
        return jsonify({"response": f"Automated Action: {context}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
