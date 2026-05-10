import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Use your API Key
API_KEY = "AIzaSyA_vKT-afxrfXW68FkNMoI1EosNVpMFGv0"
genai.configure(api_key=API_KEY.strip())

df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    file = request.files.get('file')
    if file:
        try:
            df = pd.read_csv(file)
            return jsonify({"message": "File ready!", "columns": list(df.columns)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global df
    data = request.get_json()
    user_input = data.get('message')
    
    if df is None:
        return jsonify({"response": "Please upload a CSV file first."})

    try:
        # Step 1: Automatically find the first working model in your account
        working_model_name = None
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                working_model_name = m.name
                break
        
        if not working_model_name:
            return jsonify({"response": "No working models found in your API Key."})

        # Step 2: Use the model we found
        model = genai.GenerativeModel(working_model_name)
        columns_info = ", ".join(list(df.columns))
        prompt = f"Data columns: {columns_info}. Question: {user_input}"
        
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
            
    except Exception as e:
        print(f"DEBUG ERROR: {str(e)}") # This shows the real error in terminal
        return jsonify({"response": f"Final System Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)