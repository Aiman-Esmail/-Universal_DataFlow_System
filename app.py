import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# This line tells Flask to look for HTML in the main folder, not templates
app = Flask(__name__, template_folder='.')
CORS(app)

df = None

@app.route('/')
def home():
    try:
        # Now it will find index.html in your main GitHub folder
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
        return jsonify({"message": "Success: File Loaded!", "rows": len(df), "cols": len(df.columns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    global df
    if df is None:
        return jsonify({"response": "Error: Upload data first."})
    
    null_count = df.isnull().sum().sum()
    df.fillna(method='ffill', inplace=True)
    
    report = f"Analysis: Found {len(df)} rows. Agent handled {null_count} missing values."
    return jsonify({"response": report})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
