import os
from flask import Flask, render_template, request
import pandas as pd
from data_processor import process_data

app = Flask(__name__)
# Ensure a directory for processed files
UPLOAD_FOLDER = 'static/downloads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file: return "No file", 400

    df = pd.read_csv(file)
    
    # Run the smart processing
    cleaned_df, report = process_data(df)
    
    # Save the cleaned file
    filename = "cleaned_data.csv"
    cleaned_df.to_csv(os.path.join(UPLOAD_FOLDER, filename), index=False)
    
    return render_template('index.html', 
                           report=report, 
                           download_url=filename,
                           tables=[cleaned_df.head(10).to_html(classes='data')], 
                           titles=cleaned_df.columns.values)
