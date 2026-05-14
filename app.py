import pandas as pd
import io
import os
from flask import Flask, render_template, request, send_file
from groq import Groq

app = Flask(__name__)

# Initialize Groq Client using the Environment Variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Shared variable for the cleaned data
df_cleaned = None 

@app.route('/')
def home():
    # This ensures the main page opens correctly
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():
    global df_cleaned
    
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected")

    try:
        # 1. Load Data
        df = pd.read_csv(file)
        initial_count = len(df)
        
        # 2. Automated Cleaning Logic
        df_cleaned = df.drop_duplicates().fillna(0)
        final_count = len(df_cleaned)
        
        # 3. Create Stats for AI
        stats_report = f"Initial: {initial_count} rows. Final: {final_count} rows. Removed: {initial_count - final_count} duplicates/nulls."
        
        # 4. Get AI Explanation from Llama 3.1
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional data expert. Explain the cleaning steps clearly in Arabic based on the provided statistics."},
                {"role": "user", "content": stats_report}
            ],
            model="llama-3.1-8b-instant",
        )
        ai_explanation = completion.choices[0].message.content
        
        # 5. Return all results to the webpage
        return render_template('index.html', 
                               message="Success! Data processed.", 
                               ai_response=ai_explanation)
                               
    except Exception as e:
        return render_template('index.html', message=f"Error during processing: {str(e)}")

@app.route('/download', methods=['GET'])
def download_file():
    global df_cleaned
    if df_cleaned is None:
        return "No processed data found.", 400

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='CleanData')
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Universal_DataFlow_Results.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)
