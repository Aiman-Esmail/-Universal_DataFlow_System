import pandas as pd
import io
import os
from flask import Flask, render_template, request, send_file
from groq import Groq

app = Flask(__name__)

# Initialize Groq Client
# Ensure GROQ_API_KEY is set in your Render Environment Variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df_cleaned = None 

# 1. Home Route (Fixes the 404 error)
@app.route('/')
def home():
    return render_template('index.html')

# 2. Processing Route (Cleaning + AI Report)
@app.route('/process', methods=['POST'])
def process_data():
    global df_cleaned
    
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected")

    try:
        # Load and Clean Data
        df = pd.read_csv(file)
        initial_rows = len(df)
        
        df_cleaned = df.drop_duplicates().fillna(0)
        final_rows = len(df_cleaned)
        
        # Prepare info for AI report
        stats = f"Initial rows: {initial_rows}, Final rows: {final_rows}, Removed: {initial_rows - final_rows}"
        
        # Generate AI Report using Llama 3.1
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data expert. Explain the cleaning steps clearly in Arabic based on the provided stats."},
                {"role": "user", "content": stats}
            ],
            model="llama-3.1-8b-instant",
        )
        ai_report = response.choices[0].message.content
        
        # Return everything to the UI
        return render_template('index.html', 
                               message="Processing Complete!", 
                               ai_response=ai_report)
                               
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

# 3. Download Route (Smart Export)
@app.route('/download', methods=['GET'])
def download_file():
    global df_cleaned
    if df_cleaned is None:
        return "No data processed", 400

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='CleanedData')
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Universal_DataFlow_Results.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)
