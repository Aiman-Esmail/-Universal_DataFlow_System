import os
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from imblearn.over_sampling import SMOTE

app = Flask(__name__, static_folder='static')
CORS(app)

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

# --- Step 1: Autonomous Data Agent ---
def autonomous_data_cleaner(df):
    """
    This function will automatically detect and fix data issues.
    Starting with Imbalance Detection.
    """
    actions_taken = []
    
    # Identify the target column (last column)
    target_col = df.columns[-1]
    
    # 1. Handle Class Imbalance
    class_counts = df[target_col].value_counts(normalize=True)
    if class_counts.min() < 0.40:
        smote = SMOTE(random_state=42)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_res, y_res = smote.fit_resample(X, y)
        df = pd.concat([pd.DataFrame(X_res, columns=X.columns), 
                        pd.Series(y_res, name=target_col)], axis=1)
        actions_taken.append("Fixed Class Imbalance using SMOTE.")

    # Future steps like Overfitting, Missing Values, and Outliers will be added here.
    
    return df, actions_taken

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        response = model.generate_content(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
