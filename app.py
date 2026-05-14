import pandas as pd
import io
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

# This is a placeholder for your cleaned dataframe
# In your actual code, this should be the result of your cleaning process
df_cleaned = None 

@app.route('/process', methods=['POST'])
def process_data():
    global df_cleaned
    # 1. Get the file from request
    # 2. Perform automated cleaning (Your existing logic)
    # Example:
    # df = pd.read_csv(request.files['file'])
    # df_cleaned = df.drop_duplicates().fillna(0)
    
    return render_template('index.html', message="Data cleaned successfully!")

@app.route('/download', methods=['GET'])
def download_file():
    global df_cleaned
    
    if df_cleaned is None:
        return "No data available to download", 400

    # Create an in-memory output file for the Excel data
    output = io.BytesIO()
    
    # Use Pandas with xlsxwriter engine to save the file to memory
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    
    # Reset the pointer to the beginning of the stream
    output.seek(0)
    
    # Return the file to the user's browser
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Universal_DataFlow_Results.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)
