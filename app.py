import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from data_processor import process_data
from groq import Groq 

app = Flask(__name__)

# Directory for processed files
UPLOAD_FOLDER = 'static/downloads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Groq client with the environment variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file: 
        return "No file uploaded", 400

    try:
        # Load the dataset
        df = pd.read_csv(file)
        
        # Execute the automated data cleaning pipeline
        cleaned_df, report = process_data(df)
        
        # Save the processed CSV file
        filename = "cleaned_data.csv"
        cleaned_df.to_csv(os.path.join(UPLOAD_FOLDER, filename), index=False)
        
        return render_template('index.html', 
                               report=report, 
                               download_url=filename,
                               tables=[cleaned_df.head(10).to_html(classes='data')], 
                               titles=cleaned_df.columns.values)
    except Exception as e:
        return f"Error processing file: {str(e)}", 500

# --- AI Data Assistant Route (Updated Model Name) ---
@app.route('/chat', methods=['POST'])
def chat():
    user_data = request.json
    user_query = user_data.get('query')
    
    # System instructions for the AI Agent
    system_prompt = (
        "You are an expert AI Data Assistant for the Universal DataFlow System. "
        "Your role is to explain data cleaning steps like duplicate removal, "
        "missing value imputation, and class balancing. "
        "Provide professional, concise, and technical explanations."
    )
    
    try:
        # Using the updated model name: llama-3.1-8b-instant
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return jsonify({"response": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"response": f"AI Assistant Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)