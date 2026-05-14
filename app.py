import pandas as pd
import io
import os
import base64
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_file, jsonify
from groq import Groq
from fpdf import FPDF

app = Flask(__name__)

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Global variables
df_cleaned = None
latest_ai_report = "No report generated yet."

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{encoded}"

def generate_charts(df):
    charts = []
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns[:2]
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].hist(ax=ax, color='skyblue', bins=15)
            ax.set_title(f'Distribution of {col}')
            charts.append(fig_to_base64(fig))
            plt.close(fig)
            
        if len(df.select_dtypes(include=['number']).columns) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='viridis', ax=ax)
            ax.set_title('Correlation Matrix')
            charts.append(fig_to_base64(fig))
            plt.close(fig)
    except Exception as e:
        print(f"Chart Error: {e}")
    return charts

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():
    global df_cleaned, latest_ai_report
    if 'file' not in request.files:
        return render_template('index.html', message="Error: No file uploaded.")

    file = request.files['file']
    try:
        df = pd.read_csv(file)
        df_cleaned = df.drop_duplicates().copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna("Unknown")
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        stats_summary = df_cleaned.describe(include='all').to_string()
        prompt = f"Analyze this dataset and provide 3 professional insights in English: {stats_summary}"
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional Data Analyst. Respond in English."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
        )

        latest_ai_report = response.choices[0].message.content
        charts = generate_charts(df_cleaned)
        table_html = df_cleaned.head(10).to_html(classes='table table-dark table-hover', index=False)

        return render_template('index.html', ai_response=latest_ai_report, tables=[table_html], charts=charts, message="Success!")
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

@app.route('/chat', methods=['POST'])
def chat():
    global df_cleaned
    if df_cleaned is None:
        return jsonify({"reply": "Please upload data first."})

    user_msg = request.json.get('message', '')
    system_instruction = "You are a Data Analyst assistant. Only answer questions related to your data analysis."

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_msg}
            ],
            model="llama-3.1-8b-instant",
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route('/download_pdf')
def download_pdf():
    global latest_ai_report, df_cleaned
    if df_cleaned is None: return "No data"

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="DataFlow Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        
        clean_text = latest_ai_report.encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 10, txt=clean_text)
        
        pdf_content = pdf.output(dest='S').encode('latin-1')
        return send_file(io.BytesIO(pdf_content), mimetype='application/pdf', as_attachment=True, download_name='Report.pdf')
    except Exception as e:
        return f"PDF Error: {str(e)}"

# MANDATORY: This section solves the "Port scan timeout" error
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000)) # Render provides a port dynamically
    app.run(host='0.0.0.0', port=port)
