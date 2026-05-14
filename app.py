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

    try:
        cat_cols = df.select_dtypes(include='object').columns[:2]
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            df[col].value_counts().head(5).plot(
                kind='bar', ax=ax, color='mediumseagreen', edgecolor='white'
            )
            ax.set_title(f'Top 5 Values: {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            plt.tight_layout()
            charts.append(fig_to_base64(fig))
            plt.close(fig)
    except Exception:
        pass

    try:
        numeric_cols = df.select_dtypes(include='number').columns[:4]
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            df[numeric_cols].boxplot(ax=ax)
            ax.set_title('Boxplot - Outlier Detection')
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
            else:
                df_cleaned[col] = df_cleaned[col].fillna("Unknown")

        final_stats = {"rows": len(df_cleaned)}
        real_stats = df_cleaned.describe(include='all').to_string()

        prompt = f"""
Generate a professional AI Data Analysis Report in English.
Use ONLY the following real data statistics, do not invent any numbers:

Real Data Statistics:
{real_stats}

Summary:
- Initial rows: {initial_stats['rows']}
- Final rows after cleaning: {final_stats['rows']}
- Duplicates removed: {initial_stats['duplicates']}
- Nulls per column before cleaning: {initial_stats['nulls']}
- Columns: {initial_stats['columns']}

Rules:
- Use ONLY the numbers from the real statistics above
- Do NOT invent or assume any values
- Provide output in clean English bullet points
"""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI Data Analyst. Provide reports only in English. Use ONLY the data provided, never invent numbers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
        )

        ai_report = response.choices[0].message.content
        charts = generate_charts(df_cleaned)
        preview_table = df_cleaned.head(10).to_html(
            classes='table table-hover table-bordered',
            index=True
        )

        return render_template(
            'index.html',
            message="Data Cleaned Successfully!",
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
        return jsonify({"reply": "Please upload a CSV file first before asking questions."})

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"reply": "Please enter a question."})

    try:
        real_stats = df_cleaned.describe(include='all').to_string()
        sample_data = df_cleaned.head(10).to_string()
        columns = list(df_cleaned.columns)
        shape = df_cleaned.shape

        system_prompt = f"""You are an AI Data Analyst chatbot. 
You have access to the following dataset information:

Dataset Shape: {shape[0]} rows, {shape[1]} columns
Columns: {columns}

Real Statistics:
{real_stats}

Sample Data (first 10 rows):
{sample_data}

Rules:
- Answer ONLY based on the real data provided above
- Do NOT invent or assume any values
- Be concise and clear
- Always respond in English
- If asked about something not in the data, say you don't have that information"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


@app.route('/download', methods=['GET'])
def download_file():
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
        download_name='Universal_DataFlow_Final.xlsx'
    )


if __name__ == '__main__':
    app.run(debug=True)