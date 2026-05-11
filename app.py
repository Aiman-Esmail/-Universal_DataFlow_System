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

# --- Autonomous Data Agent Logic ---
def autonomous_data_cleaner(df):
    actions_taken = []
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
        actions_taken.append("Handled Overfitting by selecting top features.")

    # 3. Handle Missing Values
    if df.isnull().values.any():
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].mean())
        actions_taken.append("Automatically filled missing values.")

    # 4. Handle Outliers
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    actions_taken.append("Handled outliers using IQR method.")
    
    return df, actions_taken

# --- Routes ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load and Clean Data
        df = pd.read_csv(file)
        cleaned_df, actions = autonomous_data_cleaner(df)
        
        # Return summary of actions and first 5 rows of cleaned data
        return jsonify({
            'message': 'Data processed successfully!',
            'actions': actions,
            'preview': cleaned_df.head().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message') or data.get('prompt')
        response = model.generate_content(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
