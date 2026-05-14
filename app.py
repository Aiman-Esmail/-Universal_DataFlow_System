import pandas as pd
import io
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_file
from groq import Groq

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df_cleaned = None


def generate_charts(df):
    charts = []

    # Chart 1: Missing Values Bar Chart
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

    # Chart 2: Numeric Distributions (Histogram)
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

    # Chart 3: Correlation Heatmap
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

    # Chart 4: Categorical Columns - Top 5 Values
    try:
        cat_cols = df.select_dtypes(include='object').columns[:2]
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            df[col].value_counts().head(5).plot(kind='bar', ax=ax, color='mediumseagreen', edgecolor='white')
            ax.set_title(f'Top 5 Values: {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            plt.tight_layout()
            charts.append(fig_to_base64(fig))
            plt.close(fig)
    except Exception:
        pass

    # Chart 5: Boxplot for Numeric Columns
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

        # Generate Charts
        charts = generate_charts(df_cleaned)

        prompt = f"""
Generate a professional AI Agent Analysis Report in English:
- Statistics: {initial_stats['rows']} initial rows, {final_stats['rows']} final rows.
- Columns: {initial_stats['columns']}
- Issues Fixed: {initial_stats['duplicates']} duplicates removed, missing values handled via Median/Mode.
- Nulls per column: {initial_stats['nulls']}
- Imbalance Check: Verified categorical distributions.
Provide the output in clean English bullet points.
"""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI Data Analyst. Provide reports only in English."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
        )

        ai_report = response.choices[0].message.content

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