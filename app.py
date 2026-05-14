import pandas as pd
import io
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

# Global variable to store the cleaned data temporarily
df_cleaned = None 

# Root route to serve the main landing page (Fixes 404 error)
@app.route('/')
def home():
    return render_template('index.html')

# Route to process uploaded data
@app.route('/process', methods=['POST'])
def process_data():
    global df_cleaned
    
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', message="No file selected")

    try:
        # Loading data (Assuming CSV for this example)
        df = pd.read_csv(file)
        
        # Automated Cleaning Logic
        df_cleaned = df.drop_duplicates().fillna(0)
        
        return render_template('index.html', message="Data processed successfully!")
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

# Route to download the cleaned Excel file
@app.route('/download', methods=['GET'])
def download_file():
    global df_cleaned
    
    if df_cleaned is None:
        return "No data available to download", 400

    output = io.BytesIO()
    
    # Using xlsxwriter to create the Excel file in memory
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned_Results')
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Universal_DataFlow_Results.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)
