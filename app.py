import pandas as pd
import numpy as np
import io
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from groq import Groq
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df_original = None
df_cleaned = None
df_ml_ready = None
latest_ai_report = ""
latest_viz_report = ""
processing_summary = {}

plt.rcParams['figure.max_open_warning'] = 0


@app.route('/health')
def health():
    return "OK", 200


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    return render_template('index.html', message="Error: File too large. Maximum size is 100MB.")


def clean_text(text):
    import re
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close('all')
    return f"data:image/png;base64,{encoded}"


def generate_charts(df):
    charts = []
    # Take an optimized small sample for visual representations
    plot_df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

    try:
        numeric_cols = plot_df.select_dtypes(include='number').columns[:2]
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(8, 3))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                plot_df[col].dropna().hist(
                    ax=ax, bins=15, color='steelblue', edgecolor='white'
                )
                ax.set_title(f'{col}')
            plt.tight_layout()
            charts.append(('Numeric Distributions', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    try:
        numeric_df = plot_df.select_dtypes(include='number')
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(7, 4))
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
            charts.append(('Correlation Heatmap', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    try:
        numeric_cols = plot_df.select_dtypes(include='number').columns[:3]
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            plot_df[numeric_cols].boxplot(ax=ax)
            ax.set_title('Boxplot - Outlier Detection')
            plt.tight_layout()
            charts.append(('Boxplot', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    try:
        cat_cols = plot_df.select_dtypes(include='object').columns[:1]
        if len(cat_cols) == 0 and len(plot_df.columns) > 0:
            cat_cols = [col for col in plot_df.columns if plot_df[col].nunique() <= 5][:1]
            
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(7, 3))
            plot_df[col].value_counts().head(5).plot(
                kind='bar', ax=ax, color='mediumseagreen', edgecolor='white'
            )
            ax.set_title(f'Class Distribution: {col}')
            plt.tight_layout()
            charts.append((f'Class Distribution: {col}', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    return charts


def process_dataframe(df):
    global processing_summary
    log = []
    summary = {}

    summary['initial_rows'] = len(df)
    summary['initial_cols'] = len(df.columns)
    
    # Optimized initial calculations to avoid high RAM spikes
    summary['nulls_before'] = {}
    summary['duplicates'] = 0
    
    if len(df) > 50000:
        log.append(f"Optimized processing enabled for large dataset ({len(df)} rows).")
        # Step 2 Optimization: Fast drop duplicates
        df = df.drop_duplicates().copy()
        summary['duplicates'] = summary['initial_rows'] - len(df)
    else:
        summary['duplicates'] = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

    # Step 1: Remove Empty Columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        log.append(f"Removed empty columns")

    # Step 3: Remove Constant Columns
    constant_cols = [col for col in df.columns if df[col].head(5000).nunique() <= 1]
    if constant_cols:
        # Verify on full data only if sample suggests it's constant
        constant_cols = [col for col in constant_cols if df[col].nunique() <= 1]
        df = df.drop(columns=constant_cols)
        log.append(f"Removed constant columns")

    # Step 4: Fix Data Types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                log.append(f"Column '{col}': converted to numeric")
            except Exception:
                pass

    # Step 5: Handle Missing Values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("Unknown")
            else:
                med_val = df[col].head(5000).median()
                df[col] = df[col].fillna(med_val)
            log.append(f"Column '{col}': treated missing values")

    # Step 6 & 7: Quick Statistical Analysis on small chunk to avoid CPU freeze
    high_corr_pairs = []
    try:
        if len(df.columns) > 1:
            sample_df = df.select_dtypes(include='number').head(5000)
            if sample_df.shape[1] >= 2:
                corr_matrix = sample_df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(col, row) for col in upper.columns for row in upper.index if upper.loc[row, col] > 0.95]
                if high_corr_pairs:
                    log.append(f"Identified potential high correlation dependencies")
    except Exception:
        pass

    # Step 8: Class Imbalance Detection and Fix (Guaranteed No Infinite Loops)
    imbalance_fixed = []
    try:
        current_columns = list(df.columns)
        for col in current_columns:
            if df[col].nunique() == 2:
                counts = df[col].value_counts()
                ratio = counts.min() / counts.max()
                if ratio < 0.5:
                    majority_class = counts.idxmax()
                    minority_class = counts.idxmin()
                    majority = df[df[col] == majority_class]
                    minority = df[df[col] == minority_class]

                    if len(df) > 30000:
                        # Downsample majority class to protect cloud memory limits
                        majority_downsampled = resample(
                            majority, replace=False, n_samples=len(minority), random_state=42
                        )
                        df = pd.concat([majority_downsampled, minority])
                        method = "Undersampling"
                    else:
                        minority_upsampled = resample(
                            minority, replace=True, n_samples=len(majority), random_state=42
                        )
                        df = pd.concat([majority, minority_upsampled])
                        method = "Oversampling"

                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    imbalance_fixed.append(f"Column '{col}': balancing applied ({method})")
                    log.append(f"Balanced class distribution for '{col}'")
                    break
    except Exception:
        pass

    summary['final_rows'] = len(df)
    summary['final_cols'] = len(df.columns)
    summary['high_corr_pairs'] = high_corr_pairs
    summary['imbalance_fixed'] = imbalance_fixed
    summary['log'] = log
    processing_summary = summary

    return df, summary, log


def create_ml_ready(df):
    # Enforce safe data constraints for ML generation step
    df_ml = df.head(30000).copy() if len(df) > 30000 else df.copy()

    for col in df_ml.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    numeric_cols = df_ml.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df_ml[numeric_cols] = scaler.fit_transform(df_ml[numeric_cols])

    return df_ml


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_data():
    global df_original, df_cleaned, df_ml_ready
    global latest_ai_report, latest_viz_report, processing_summary

    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected")

    try:
        df_original = pd.read_csv(file)
        df_cleaned, summary, log = process_dataframe(df_original.copy())
        df_ml_ready = create_ml_ready(df_cleaned)

        preprocessing_prompt = f"""
Write a professional data preprocessing report.
Use plain sentences only without markdown formatting.
Sections: 1. Overview  2. Steps Applied  3. Quality Assessment

Summary:
Initial rows: {summary['initial_rows']} | Columns: {summary['initial_cols']}
Final rows: {summary['final_rows']} | Columns: {summary['final_cols']}
Duplicates removed: {summary['duplicates']}
"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional Data Analyst. Respond only in plain text paragraphs."},
                {"role": "user", "content": preprocessing_prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        latest_ai_report = clean_text(response.choices[0].message.content)

        viz_prompt = f"""
Write a brief data visualization report summary. No markdown formatting.
Explain the value of distribution histogram graphs and correlation matrices for pipeline analysis.
"""

        viz_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a Data Expert. Respond only in plain text paragraphs."},
                {"role": "user", "content": viz_prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        latest_viz_report = clean_text(viz_response.choices[0].message.content)

        charts = generate_charts(df_cleaned)
        preview_table = df_cleaned.head(10).to_html(
            classes='table table-hover table-bordered', index=True
        )

        return render_template(
            'index.html',
            message="Data Processed Successfully!",
            ai_response=latest_ai_report,
            viz_response=latest_viz_report,
            tables=[preview_table],
            charts=charts,
            initial_rows=summary['initial_rows'],
            final_rows=summary['final_rows'],
            duplicates=summary['duplicates'],
            columns=list(df_cleaned.columns),
            preprocessing_log=log
        )

    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")


@app.route('/chat', methods=['POST'])
def chat():
    global df_cleaned

    if df_cleaned is None:
        return jsonify({"reply": "Please upload a CSV file first before asking questions."})

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"reply": "Please enter a question."})

    try:
        system_prompt = f"""You are a strict Data Analysis Assistant.
Dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns.
Columns: {list(df_cleaned.columns[:10])}
Answer short and concise without using markdown or asterisks in the same language the user asks."""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
        )
        reply = clean_text(response.choices[0].message.content)
        return jsonify({ "reply": reply })

    except Exception as e:
        return jsonify({ "reply": f"Error: {str(e)}" })


@app.route('/download_csv', methods=['GET'])
def download_csv():
    global df_cleaned
    if df_cleaned is None:
        return render_template('index.html', message="No data available")
    output = io.StringIO()
    df_cleaned.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='Cleaned_Data.csv'
    )


@app.route('/download_ml', methods=['GET'])
def download_ml():
    global df_ml_ready
    if df_ml_ready is None:
        return render_template('index.html', message="No data available")
    output = io.StringIO()
    df_ml_ready.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='ML_Ready_Data.csv'
    )


@app.route('/download_excel', methods=['GET'])
def download_excel():
    global df_cleaned
    if df_cleaned is None:
        return render_template('index.html', message="No data available")
    
    export_df = df_cleaned.head(30000) if len(df_cleaned) > 30000 else df_cleaned
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Cleaned_Data.xlsx'
    )


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    global df_cleaned, latest_ai_report, processing_summary
    if df_cleaned is None:
        return render_template('index.html', message="No data available")

    try:
        output = io.BytesIO()
        doc = SimpleDocTemplate(
            output, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CTitle', parent=styles['Title'], fontSize=20, spaceAfter=20)
        normal_style = ParagraphStyle('CNormal', parent=styles['Normal'], fontSize=10, spaceAfter=6, leading=14)

        story = [
            Paragraph("Universal DataFlow System Report", title_style),
            Spacer(1, 15),
            Paragraph(f"Initial Dataset Rows: {processing_summary.get('initial_rows')}", normal_style),
            Paragraph(f"Processed Dataset Rows: {processing_summary.get('final_rows')}", normal_style),
            Spacer(1, 15),
            Paragraph("AI Preprocessing Analysis Summary:", styles['Heading3']),
            Paragraph(latest_ai_report, normal_style)
        ]
        doc.build(story)
        output.seek(0)
        return send_file(output, mimetype='application/pdf', as_attachment=True, download_name='DataFlow_Report.pdf')
    except Exception as e:
        return f"PDF Generation Error: {str(e)}"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
