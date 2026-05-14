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
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df_original = None
df_cleaned = None
latest_ai_report = "No report generated yet."
latest_viz_report = "No visualization report yet."

plt.rcParams['figure.max_open_warning'] = 0


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

    # Chart 1: Distributions
    try:
        numeric_cols = df.select_dtypes(include='number').columns[:2]
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(8, 3))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                df[col].dropna().hist(ax=ax, bins=15, color='steelblue', edgecolor='white')
                ax.set_title(f'{col}')
            plt.tight_layout()
            charts.append(('Distributions', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    # Chart 2: Correlation Heatmap
    try:
        numeric_df = df.select_dtypes(include='number')
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

    # Chart 3: Boxplot
    try:
        numeric_cols = df.select_dtypes(include='number').columns[:3]
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            df[numeric_cols].boxplot(ax=ax)
            ax.set_title('Boxplot - Outlier Detection')
            plt.tight_layout()
            charts.append(('Boxplot', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    # Chart 4: Categorical Top 5
    try:
        cat_cols = df.select_dtypes(include='object').columns[:1]
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(7, 3))
            df[col].value_counts().head(5).plot(
                kind='bar', ax=ax, color='mediumseagreen', edgecolor='white'
            )
            ax.set_title(f'Top 5: {col}')
            plt.tight_layout()
            charts.append((f'Top Values: {col}', fig_to_base64(fig)))
            plt.close('all')
    except Exception:
        plt.close('all')

    return charts


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_data():
    global df_original, df_cleaned, latest_ai_report, latest_viz_report

    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected")

    try:
        df_original = pd.read_csv(file)

        initial_stats = {
            "rows": len(df_original),
            "columns": list(df_original.columns),
            "nulls": df_original.isnull().sum().to_dict(),
            "duplicates": int(df_original.duplicated().sum())
        }

        df_cleaned = df_original.drop_duplicates().copy()

        preprocessing_log = []
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                mode_val = df_cleaned[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                nulls_count = df_cleaned[col].isnull().sum()
                if nulls_count > 0:
                    preprocessing_log.append(
                        f"Column '{col}': filled {nulls_count} nulls with mode '{fill_val}'"
                    )
                df_cleaned[col] = df_cleaned[col].fillna(fill_val)
            elif pd.api.types.is_numeric_dtype(df_cleaned[col]):
                med_val = df_cleaned[col].median()
                nulls_count = df_cleaned[col].isnull().sum()
                if nulls_count > 0:
                    preprocessing_log.append(
                        f"Column '{col}': filled {nulls_count} nulls with median {med_val:.2f}"
                    )
                df_cleaned[col] = df_cleaned[col].fillna(med_val)
            else:
                df_cleaned[col] = df_cleaned[col].fillna("Unknown")

        final_stats = {"rows": len(df_cleaned)}
        real_stats = df_cleaned.describe(include='all').to_string()

        # AI Preprocessing Report
        preprocessing_prompt = f"""
Generate a professional Data Preprocessing Report in English.
Use ONLY the following real data, do not invent numbers.

Initial rows: {initial_stats['rows']}
Final rows after cleaning: {final_stats['rows']}
Duplicates removed: {initial_stats['duplicates']}
Nulls per column before cleaning: {initial_stats['nulls']}
Columns: {initial_stats['columns']}
Preprocessing actions: {preprocessing_log}

Real Statistics after cleaning:
{real_stats}

Provide:
1. Data Overview
2. Preprocessing Steps Applied
3. Data Quality Assessment
4. Key Findings
Use clean English bullet points only.
"""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI Data Analyst. Use ONLY the data provided, never invent numbers."
                },
                {"role": "user", "content": preprocessing_prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        latest_ai_report = response.choices[0].message.content

        # AI Visualization Report
        viz_prompt = f"""
Generate a professional Data Visualization Report in English.
Based on this real dataset:

{real_stats}

Columns: {initial_stats['columns']}

Explain what the following charts show:
1. Distribution charts for numeric columns
2. Correlation heatmap
3. Boxplot for outlier detection
4. Top values for categorical columns

Use ONLY real statistics. Clean English bullet points.
"""

        viz_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI Data Visualization Expert. Use ONLY the data provided."
                },
                {"role": "user", "content": viz_prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        latest_viz_report = viz_response.choices[0].message.content

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
            initial_rows=initial_stats['rows'],
            final_rows=final_stats['rows'],
            duplicates=initial_stats['duplicates'],
            columns=initial_stats['columns'],
            preprocessing_log=preprocessing_log
        )

    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")


@app.route('/chat', methods=['POST'])
def chat():
    global df_cleaned

    if df_cleaned is None:
        return jsonify({"reply": "Please upload a CSV file first."})

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"reply": "Please enter a question."})

    try:
        real_stats = df_cleaned.describe(include='all').to_string()
        sample_data = df_cleaned.head(5).to_string()

        system_prompt = f"""You are an AI Data Analyst chatbot.
Dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns
Columns: {list(df_cleaned.columns)}
Statistics:
{real_stats}
Sample:
{sample_data}
Rules:
- Answer ONLY based on real data
- Never invent values
- Be concise and clear
- Respond in English"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
        )
        return jsonify({"reply": response.choices[0].message.content})

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
    global df_cleaned, latest_ai_report, latest_viz_report

    if df_cleaned is None:
        return render_template('index.html', message="No data available")

    try:
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Universal DataFlow System - Full Report", styles['Title']))
        story.append(Spacer(1, 20))

        # Preprocessing Report
        story.append(Paragraph("Preprocessing Report", styles['Heading1']))
        story.append(Spacer(1, 10))
        for line in latest_ai_report.split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), styles['Normal']))
                story.append(Spacer(1, 5))

        story.append(Spacer(1, 20))

        # Visualization Report
        story.append(Paragraph("Visualization Report", styles['Heading1']))
        story.append(Spacer(1, 10))
        for line in latest_viz_report.split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), styles['Normal']))
                story.append(Spacer(1, 5))

        story.append(Spacer(1, 20))

        # Stats Table
        story.append(Paragraph("Data Statistics Summary", styles['Heading1']))
        story.append(Spacer(1, 10))

        stats_df = df_cleaned.describe(include='all').reset_index()
        table_data = [list(stats_df.columns)]
        for _, row in stats_df.iterrows():
            table_data.append([
                str(round(v, 2)) if isinstance(v, float) else str(v)
                for v in row
            ])

        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
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