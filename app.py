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
from groq import Groq
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df_original = None
df_cleaned = None
df_ml_ready = None
latest_ai_report = ""
latest_viz_report = ""
processing_summary = {}

plt.rcParams['figure.max_open_warning'] = 0


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

    try:
        numeric_cols = df.select_dtypes(include='number').columns[:2]
        if len(numeric_cols) > 0:
            sample = df[numeric_cols].sample(min(5000, len(df)), random_state=42)
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(8, 3))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                sample[col].dropna().hist(ax=ax, bins=15, color='steelblue', edgecolor='white')
                ax.set_title(f'{col}')
            plt.tight_layout()
            charts.append(('Numeric Distributions', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    try:
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] >= 2:
            sample = numeric_df.sample(min(5000, len(numeric_df)), random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(
                sample.corr(),
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
        numeric_cols = df.select_dtypes(include='number').columns[:3]
        if len(numeric_cols) > 0:
            sample = df[numeric_cols].sample(min(5000, len(df)), random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            sample.boxplot(ax=ax)
            ax.set_title('Boxplot - Outlier Detection')
            plt.tight_layout()
            charts.append(('Boxplot', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    try:
        cat_cols = df.select_dtypes(include='object').columns[:1]
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(7, 3))
            df[col].value_counts().head(5).plot(
                kind='bar', ax=ax, color='mediumseagreen', edgecolor='white'
            )
            ax.set_title(f'Class Distribution: {col}')
            plt.tight_layout()
            charts.append((f'Class Distribution: {col}', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    try:
        if processing_summary.get('nulls_before'):
            nulls = pd.Series(processing_summary['nulls_before'])
            nulls = nulls[nulls > 0]
            if not nulls.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                nulls.plot(kind='bar', ax=ax, color='tomato', edgecolor='white')
                ax.set_title('Missing Values Before Cleaning')
                plt.tight_layout()
                charts.append(('Missing Values', fig_to_base64(fig)))
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
    summary['nulls_before'] = df.isnull().sum().to_dict()
    summary['duplicates'] = int(df.duplicated().sum())

    # Step 1: Remove Empty Columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        log.append(f"Removed {len(empty_cols)} completely empty columns: {empty_cols}")

    # Step 2: Remove Duplicates
    if summary['duplicates'] > 0:
        df = df.drop_duplicates().copy()
        log.append(f"Removed {summary['duplicates']} duplicate rows")

    # Step 3: Remove Constant Columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        log.append(f"Removed {len(constant_cols)} constant columns: {constant_cols}")

    # Step 4: Fix Data Types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                log.append(f"Column '{col}': converted from text to numeric")
            except Exception:
                pass

    # Step 5: Handle Missing Values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if df[col].dtype == 'object':
                mode_val = df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                log.append(f"Column '{col}': filled {null_count} missing values with most common value '{fill_val}'")
            elif pd.api.types.is_numeric_dtype(df[col]):
                med_val = df[col].median()
                df[col] = df[col].fillna(med_val)
                log.append(f"Column '{col}': filled {null_count} missing values with median {med_val:.2f}")

    # Step 6: High Correlation Detection on sample
    high_corr_pairs = []
    try:
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] >= 2:
            sample_df = numeric_df.sample(min(5000, len(numeric_df)), random_state=42)
            corr_matrix = sample_df.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr = [
                (col, row, round(upper.loc[row, col], 2))
                for col in upper.columns
                for row in upper.index
                if upper.loc[row, col] > 0.95
            ]
            if high_corr:
                high_corr_pairs = high_corr
                log.append(f"Detected {len(high_corr)} highly correlated column pairs (above 0.95): {[(p[0], p[1]) for p in high_corr]}")
    except Exception:
        pass

    # Step 7: Low Variance Detection
    try:
        numeric_df = df.select_dtypes(include='number')
        low_var_cols = [col for col in numeric_df.columns if numeric_df[col].std() < 0.01]
        if low_var_cols:
            log.append(f"Detected {len(low_var_cols)} low variance columns: {low_var_cols}")
    except Exception:
        pass

    # Step 8: Class Imbalance Detection and Fix
    imbalance_fixed = []
    try:
        for col in df.columns:
            if df[col].nunique() == 2:
                counts = df[col].value_counts()
                ratio = counts.min() / counts.max()
                if ratio < 0.5:
                    majority_class = counts.idxmax()
                    minority_class = counts.idxmin()
                    majority = df[df[col] == majority_class]
                    minority = df[df[col] == minority_class]

                    if len(df) > 50000:
                        majority_downsampled = resample(
                            majority,
                            replace=False,
                            n_samples=min(len(minority) * 2, len(majority)),
                            random_state=42
                        )
                        df = pd.concat([majority_downsampled, minority])
                        method = "Undersampling"
                    else:
                        minority_upsampled = resample(
                            minority,
                            replace=True,
                            n_samples=len(majority),
                            random_state=42
                        )
                        df = pd.concat([majority, minority_upsampled])
                        method = "Oversampling"

                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    imbalance_fixed.append(
                        f"Column '{col}': applied {method}, final rows: {len(df)}"
                    )
                    log.append(
                        f"Column '{col}': imbalance ratio {ratio:.2f}, applied {method}"
                    )
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
    df_ml = df.copy()

    for col in df_ml.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    numeric_cols = df_ml.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        if len(df_ml) > 50000:
            sample = df_ml[numeric_cols].sample(10000, random_state=42)
            scaler.fit(sample)
        else:
            scaler.fit(df_ml[numeric_cols])
        df_ml[numeric_cols] = scaler.transform(df_ml[numeric_cols])

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

        real_stats = df_cleaned.describe(include='all').to_string()

        preprocessing_prompt = f"""
Write a professional data preprocessing report.
Use plain sentences only. No bullet symbols, no asterisks, no hashtags, no markdown.
Use simple numbered sections: 1. Overview  2. Steps Applied  3. Quality Assessment  4. Key Findings

Use ONLY this real data, do not invent numbers:

Initial rows: {summary['initial_rows']}
Initial columns: {summary['initial_cols']}
Final rows: {summary['final_rows']}
Final columns: {summary['final_cols']}
Duplicates removed: {summary['duplicates']}
Missing values before cleaning: {summary['nulls_before']}
High correlation pairs detected: {summary['high_corr_pairs']}
Imbalance fixed: {summary['imbalance_fixed']}
All steps applied: {log}

Statistics after cleaning:
{real_stats}
"""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional Data Analyst. Write clean reports in plain English. Never use asterisks, hashtags, bullet symbols, or any markdown. Use numbered sections and plain paragraphs only."
                },
                {"role": "user", "content": preprocessing_prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        latest_ai_report = clean_text(response.choices[0].message.content)

        viz_prompt = f"""
Write a professional data visualization report explaining what the charts show.
Use plain sentences only. No bullet symbols, no asterisks, no hashtags, no markdown.
Use simple numbered sections.

Dataset columns: {list(df_cleaned.columns)}
Statistics: {real_stats}

Explain clearly:
1. What the distribution charts show
2. What the correlation heatmap reveals
3. What the boxplot indicates about outliers
4. What the class distribution shows
"""

        viz_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a Data Visualization Expert. Write clean reports in plain English. Never use asterisks, hashtags, bullet symbols, or markdown. Use numbered sections and plain paragraphs only."
                },
                {"role": "user", "content": viz_prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        latest_viz_report = clean_text(viz_response.choices[0].message.content)

        charts = generate_charts(df_cleaned)
        preview_table = df_cleaned.head(10).to_html(
            classes='table table-hover table-bordered',
            index=True
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
    global df_cleaned, processing_summary

    if df_cleaned is None:
        return jsonify({"reply": "Please upload a CSV file first before asking questions."})

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"reply": "Please enter a question."})

    try:
        real_stats = df_cleaned.describe(include='all').to_string()
        sample_data = df_cleaned.head(5).to_string()

        system_prompt = f"""You are a strict Data Analysis Assistant.
Your ONLY job is to answer questions about the uploaded dataset.

Dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns
Columns: {list(df_cleaned.columns)}

Real Statistics:
{real_stats}

Sample Data:
{sample_data}

Processing Steps Applied:
{processing_summary.get('log', [])}

Imbalance Fixed:
{processing_summary.get('imbalance_fixed', [])}

High Correlation Detected:
{processing_summary.get('high_corr_pairs', [])}

STRICT RULES:
1. Answer in the SAME language the user uses
2. ONLY answer questions about this dataset
3. Topics allowed: cleaning steps, statistics, columns, values, imbalance, correlation, overfitting, preprocessing, distributions, missing values, duplicates
4. If user asks anything off-topic respond in their language: "I am a Data Analysis Assistant. I can only answer questions about your uploaded dataset."
5. Never invent values
6. Be concise and professional
7. Never use asterisks or markdown symbols in responses"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
        )
        reply = clean_text(response.choices[0].message.content)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


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
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Cleaned_Data.xlsx'
    )


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    global df_cleaned, latest_ai_report, latest_viz_report, processing_summary

    if df_cleaned is None:
        return render_template('index.html', message="No data available")

    try:
        output = io.BytesIO()
        doc = SimpleDocTemplate(
            output,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=20,
            spaceAfter=20,
            textColor=colors.HexColor('#1a1a2e')
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#667eea')
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leading=16
        )
        log_style = ParagraphStyle(
            'LogStyle',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=4,
            leftIndent=20,
            textColor=colors.HexColor('#2d3748')
        )

        story = []

        story.append(Paragraph("Universal DataFlow System", title_style))
        story.append(Paragraph("Automated Data Processing Report", styles['Heading2']))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Processing Summary", heading_style))
        summary_data = [
            ['Metric', 'Value'],
            ['Initial Rows', str(processing_summary.get('initial_rows', 'N/A'))],
            ['Final Rows', str(processing_summary.get('final_rows', 'N/A'))],
            ['Initial Columns', str(processing_summary.get('initial_cols', 'N/A'))],
            ['Final Columns', str(processing_summary.get('final_cols', 'N/A'))],
            ['Duplicates Removed', str(processing_summary.get('duplicates', 0))],
            ['Imbalance Fixed', str(len(processing_summary.get('imbalance_fixed', [])))],
            ['High Correlation Pairs', str(len(processing_summary.get('high_corr_pairs', [])))],
        ]
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWHEIGHT', (0, 0), (-1, -1), 25),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("Processing Steps Applied", heading_style))
        log = processing_summary.get('log', [])
        if log:
            for i, step in enumerate(log, 1):
                story.append(Paragraph(f"{i}. {step}", log_style))
        else:
            story.append(Paragraph("No issues found. Data was already clean.", normal_style))
        story.append(Spacer(1, 15))

        story.append(Paragraph("AI Preprocessing Analysis", heading_style))
        for line in latest_ai_report.split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), normal_style))
        story.append(Spacer(1, 15))

        story.append(Paragraph("AI Visualization Analysis", heading_style))
        for line in latest_viz_report.split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), normal_style))
        story.append(Spacer(1, 15))

        story.append(Paragraph("Data Statistics After Cleaning", heading_style))
        stats_df = df_cleaned.describe(include='all').reset_index()
        max_cols = min(len(stats_df.columns), 7)
        stats_df = stats_df.iloc[:, :max_cols]
        table_data = [[str(c) for c in stats_df.columns]]
        for _, row in stats_df.iterrows():
            table_data.append([
                str(round(v, 2)) if isinstance(v, float) else str(v)
                for v in row
            ])
        col_width = 6.5 * inch / max_cols
        t = Table(table_data, colWidths=[col_width] * max_cols)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(t)

        doc.build(story)
        output.seek(0)

        return send_file(
            output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='DataFlow_Full_Report.pdf'
        )

    except Exception as e:
        return f"PDF Error: {str(e)}"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)