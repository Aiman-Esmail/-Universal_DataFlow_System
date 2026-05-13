import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from data_processor import process_data
from groq import Groq 

app = Flask(__name__)

# Directory settings
UPLOAD_FOLDER = 'static/downloads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Groq client (Make sure GROQ_API_KEY is set in your environment)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file: return "No file", 400

    df = pd.read_csv(file)
    
    # Run the smart processing
    cleaned_df, report = process_data(df)
    
    # Save the cleaned file
    filename = "cleaned_data.csv"
    cleaned_df.to_csv(os.path.join(UPLOAD_FOLDER, filename), index=False)
    
    return render_template('index.html', 
                           report=report, 
                           download_url=filename,
                           tables=[cleaned_df.head(10).to_html(classes='data')], 
                           titles=cleaned_df.columns.values)

# --- New Route for AI Chatbot Assistant ---
@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    
    # System context for the AI Assistant (Explainable AI)
    context = (
        "You are an AI Data Assistant for the Universal DataFlow System. "
        "The system performs: duplicate removal, missing value imputation, and class balancing. "
        "Answer user questions professionally and concisely based on data cleaning logic."
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return jsonify({"response": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
