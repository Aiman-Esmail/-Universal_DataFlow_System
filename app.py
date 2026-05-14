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
latest_ai_report = ""

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{encoded}"

def generate_charts(df):
    charts = []
    try:
        # Missing Values
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            nulls.plot(kind='bar', ax=ax, color='tomato')
            ax.set_title('Missing Values Report')
            plt.tight_layout()
            charts.append(fig_to_base64(fig))
            plt.close(fig)
            
        # Correlation
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix')
            plt.tight_layout()
            charts.append(fig_to_base64(fig))
            plt.close(fig)
    except: pass
    return charts

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():
    global df_cleaned, latest_ai_report
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")

    file = request.files['file']
    try:
        df = pd.read_csv(file)
        initial_rows = len(df)
        initial_dupes = int(df.duplicated().sum())
        
        # Preprocessing
        df_cleaned = df.drop_duplicates().copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "N/A")
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        stats_summary = df_cleaned.describe(include='all').to_string()

        prompt = f"Generate a professional Data Analysis Report for a dataset with {len(df_cleaned)} rows. Summarize preprocessing and key insights in English bullet points. Data stats: {stats_summary}"

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional Data Analyst. Report in English."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
        )

        latest_ai_report = response.choices[0].message.content
        charts = generate_charts(df_cleaned)
        table_html = df_cleaned.head(10).to_html(classes='table table-striped', index=False)

        return render_template('index.html', ai_response=latest_ai_report, tables=[table_html], charts=charts, message="Success!")
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

@app.route('/chat', methods=['POST'])
def chat():
    global df_cleaned
    if df_cleaned is None: return jsonify({"reply": "Upload data first."})
    user_msg = request.json.get('message', '')
    
    system_instruction = f"""You are a STRICT Data Science Assistant. 
    1. ONLY answer data-related questions. 
    2. If asked about food/recipes/anything else, say: 'I am a Data Analyst. I cannot answer non-data related questions.'"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_msg}],
            model="llama-3.1-8b-instant",
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"reply": f"AI Error: {str(e)}"})

# --- NEW: PDF REPORT GENERATION ---
@app.route('/download_pdf')
def download_pdf():
    global latest_ai_report, df_cleaned
    if df_cleaned is None: return "No data"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Universal DataFlow - Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=latest_ai_report.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf_output = io.BytesIO()
    pdf_str = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_str)
    pdf_output.seek(0)
    
    return send_file(pdf_output, mimetype='application/pdf', as_attachment=True, download_name='Analysis_Report.pdf')

@app.route('/download_csv')
def download_csv():
    global df_cleaned
    output = io.StringIO()
    df_cleaned.to_csv(output, index=False)
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='Cleaned_Data.csv')

if __name__ == '__main__':
    app.run(debug=True)
