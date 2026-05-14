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

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Global variable to store the cleaned data across routes
df_cleaned = None

def generate_charts(df):
    charts = []
    try:
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            nulls.plot(kind='bar', ax=ax, color='tomato')
            ax.set_title('Missing Values per Column')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Count')
            plt.tight_layout()
            charts.append(fig_to_base64(fig))
            plt.close(fig)
    except Exception:
        pass

    try:
        numeric_cols = df.select_dtypes(include='number').columns[:4]
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                df[col].dropna().hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
                ax.set_title(f'Distribution: {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
            plt.tight_layout()
            charts.append(fig_to_base64(fig))
            plt.close(fig)
    except Exception:
        pass

    try:
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                ax=ax,
                linewidths=0.5
            )
            ax.set_title('Correlation Heatmap')
            plt.tight_layout()
            charts.append(fig_to_base64(fig))
            plt.close(fig)
    except Exception:
        pass

    return charts

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{encoded}"

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
        df = pd.read_csv(file)
        initial_stats = {
            "rows": len(df),
            "columns": list(df.columns),
            "nulls": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum())
        }

        # Data Preprocessing Logic
        df_cleaned = df.drop_duplicates().copy()

        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                mode_val = df_cleaned[col].mode()
                df_cleaned[col] = df_cleaned[col].fillna(
                    mode_val[0] if not mode_val.empty else "Unknown"
                )
            elif pd.api.types.is_numeric_dtype(df_cleaned[col]):
                df_cleaned[col] = df_cleaned[col].fillna(
                    df_cleaned[col].median()
                )

        final_stats = {"rows": len(df_cleaned)}
        real_stats = df_cleaned.describe(include='all').to_string()

        # Detailed Preprocessing Report Prompt
        prompt = f"""
Generate a professional AI Data Analysis and Preprocessing Report.
Dataset Summary:
- Initial Records: {initial_stats['rows']}
- Final Records after removing {initial_stats['duplicates']} duplicates: {final_stats['rows']}
- Missing values handled in columns: {initial_stats['columns']}

Stats after processing:
{real_stats}

Requirements:
1. Explain the preprocessing steps taken (Duplicate removal, Null handling).
2. Summarize key statistical insights.
3. Output MUST be in English and use bullet points.
"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional Data Scientist. Provide detailed reports in English only."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
        )

        ai_report = response.choices[0].message.content
        charts = generate_charts(df_cleaned)
        preview_table = df_cleaned.head(10).to_html(classes='table table-striped', index=True)

        return render_template(
            'index.html',
            message="Data Processed and Cleaned Successfully!",
            ai_response=ai_report,
            tables=[preview_table],
            charts=charts
        )

    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

@app.route('/chat', methods=['POST'])
def chat():
    global df_cleaned
    if df_cleaned is None:
        return jsonify({"reply": "Please upload a CSV file first."})

    user_message = request.json.get('message', '')
    try:
        real_stats = df_cleaned.describe(include='all').to_string()
        system_prompt = f"""You are an AI Analyst for the Universal DataFlow System.
        Dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns.
        Stats: {real_stats}
        Rules: Respond in the user's language (Arabic, English, or German). Be data-driven."""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"reply": f"AI Error: {str(e)}"})

# NEW: Route for CSV Download as requested
@app.route('/download_csv', methods=['GET'])
def download_csv():
    global df_cleaned
    if df_cleaned is None:
        return render_template('index.html', message="No data available to download.")

    output = io.StringIO()
    df_cleaned.to_csv(output, index=False)
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name='Cleaned_Data_Universal.csv'
    )

# Keeping Excel download option as well
@app.route('/download_excel', methods=['GET'])
def download_excel():
    global df_cleaned
    if df_cleaned is None:
        return render_template('index.html', message="No data available")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned_Data')

    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Cleaned_Data_Universal.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)
