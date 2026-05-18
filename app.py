import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_universal_dataflow'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
        
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        session['current_file'] = file.filename
        return redirect(url_for('process_data'))

@app.route('/process_data')
def process_data():
    filename = session.get('current_file')
    if not filename:
        return redirect(url_for('index'))
        
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
            
        # Dynamic Data Cleaning
        df = df.dropna()
        
        shape_info = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns after dynamic cleaning."
        html_preview = df.head(10).to_html(classes='table table-striped table-hover border text-center')
        
        return render_template('dashboard.html', filename=filename, shape=shape_info, preview=html_preview)
        
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}"

@app.route('/download_pdf')
def download_pdf():
    filename = session.get('current_file')
    if not filename:
        return "No active dataset found to generate PDF", 400
        
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
            
        df = df.dropna()
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Dataset_Report.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = styles['Title']
        title_style.fontName = 'Helvetica-Bold'
        
        normal_style = styles['Normal']
        normal_style.fontName = 'Helvetica'
        
        story.append(Paragraph("Universal DataFlow System - Analysis Report", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"<b>Processed File:</b> {filename}", normal_style))
        story.append(Paragraph(f"<b>Dataset Shape:</b> {df.shape[0]} rows and {df.shape[1]} columns.", normal_style))
        story.append(Spacer(1, 15))
        
        heading_style = styles['Heading3']
        heading_style.fontName = 'Helvetica-Bold'
        story.append(Paragraph("<b>Dataset Preview (Top 5 Rows):</b>", heading_style))
        story.append(Spacer(1, 10))
        
        sub_df = df.iloc[:5, :6]
        table_data = [list(sub_df.columns)] + sub_df.values.tolist()
        table_data = [[str(item) for item in row] for row in table_data]
        
        pdf_table = Table(table_data)
        pdf_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        
        story.append(pdf_table)
        doc.build(story)
        
        return send_file(pdf_path, as_attachment=True)
        
    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500

@app.route('/clear')
def clear():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)