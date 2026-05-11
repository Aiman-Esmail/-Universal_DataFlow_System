import os
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from imblearn.over_sampling import SMOTE

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Configuration: Gemini API ---
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

# --- STEP 1, 2 & 3: Autonomous Data Agent Logic ---
def autonomous_data_cleaner(df):
    """
    This function automatically detects and fixes data issues.
    Handles: Imbalance, Overfitting, and Missing Values.
    """
    actions_taken = []
    
    # Identify the target column (assumed to be the last one)
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

    # 2. Handle Potential Overfitting
    if len(df.columns) > (len(df) * 0.1):
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        num_features = min(10, len(df.columns))
        top_features = correlations.index[:num_features]
        df = df[top_features]
        actions_taken.append("Handled Overfitting by selecting most relevant features.")

    # 3. Handle Missing Values
    if df.isnull().values.any():
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    # Fill text columns with the most frequent value (Mode)
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    # Fill numeric columns with the average (Mean)
                    df[col] = df[col].fillna(df[col].mean())
        actions_taken.append("Automatically filled missing values (Mean/Mode).")
    
    return df, actions_taken

# --- Routes ---
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
        user_message = data.get('message') or data.get('prompt') or data.get('text')
        
        if not user_message:
            return jsonify({'error': 'No input detected.'}), 400

        response = model.generate_content(user_message)
        
        if response and response.text:
            return jsonify({'response': response.text})
        else:
            return jsonify({'error': 'AI could not generate a response.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
