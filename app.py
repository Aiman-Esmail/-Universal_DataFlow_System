import pandas as pd
import io
import os
from flask import Flask, render_template, request, send_file
from groq import Groq

app = Flask(__name__)

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df_cleaned = None 

@app.route('/')
def home():
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
        # Load Original Data
        df = pd.read_csv(file)
        
        # 1. Analyze Data for AI
        null_counts = df.isnull().sum().to_dict()
        duplicates = int(df.duplicated().sum())
        columns = list(df.columns)
        
        # 2. Automated Cleaning Process
        # Handling missing values (using median/mode logic equivalent)
        df_cleaned = df.drop_duplicates()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown")
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        # 3. Check for Imbalance (for categorical columns)
        imbalance_info = ""
        cat_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in cat_cols:
            counts = df_cleaned[col].value_counts(normalize=True)
            if counts.max() > 0.8: # Threshold for imbalance
                imbalance_info += f"Column '{col}' is imbalanced ({round(counts.max()*100, 2)}% in one class). "

        # 4. Prompt for AI Analysis Report (In English)
        prompt = f"""
        Analyze this data cleaning task and provide a professional bullet-point report in English:
        - Original Columns: {columns}
        - Missing values found: {null_counts}
        - Duplicates removed: {duplicates}
        - Imbalance Check: {imbalance_info if imbalance_info else 'Dataset is balanced.'}
        - Actions taken: Fixed missing values using Median/Mode imputation and removed duplicates to ensure data integrity.
        """
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional Data Analyst. Provide a concise AI Agent Analysis Report in English with bullet points, matching the style in the user's previous documentation."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        ai_report = response.choices[0].message.content
        
        # 5. Get first 10 rows for preview (as seen in image)
        preview_data = df_cleaned.head(10).to_html(classes='table table-striped', index=True)
        
        return render_template('index.html', 
                               message="Success! Data processed.", 
                               ai_response=ai_report,
                               tables=[preview_data])
                               
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

@app.route('/download', methods=['GET'])
def download_file():
    global df_cleaned
    if df_cleaned is None: return "No data", 400
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name='Cleaned_Dataset.xlsx')

if __name__ == '__main__':
    app.run(debug=True)
