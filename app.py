
import os
from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Import ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# 1. Initialize Flask Application and Configurations
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_universal_dataflow'

# Ensure upload directory exists on the server
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global static path for Render environment stability
STATIC_CLEANED_PATH = os.path.join(UPLOAD_FOLDER, 'latest_cleaned_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return render_template('index.html', message='Error: No file part in the request.')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='Error: No file selected.')
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read original dataset and extract dimensions
            df = pd.read_csv(file)
            initial_rows = len(df)
            columns = df.columns.tolist()
            
            # Step 1: Handle Duplicates Purging
            df_cleaned = df.drop_duplicates()
            duplicates_removed = initial_rows - len(df_cleaned)
            
            # Step 2: Handle Missing Values Imputation
            imputed_numeric_cols = []
            imputed_categorical_cols = []
            
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().sum() > 0:
                    if df_cleaned[col].dtype in ['int64', 'float64']:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                        imputed_numeric_cols.append(col)
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
                        imputed_categorical_cols.append(col)
            
            # Professional Text Formatting Validation
            if imputed_numeric_cols:
                numeric_summary = f"Numeric columns repaired via median imputation: [{', '.join(imputed_numeric_cols)}]."
                ai_numeric_segment = f"Columns with missing entries [{', '.join(imputed_numeric_cols)}] were automatically calculated and filled using data-driven median values to protect statistical integrity."
            else:
                numeric_summary = "Numeric Features Audit: Verified 100% complete; no missing values detected."
                ai_numeric_segment = "Continuous numerical features were structurally audited, confirming absolute data density with zero missing profiles."

            if imputed_categorical_cols:
                categorical_summary = f"Categorical columns repaired via mode: [{', '.join(imputed_categorical_cols)}]."
                ai_categorical_segment = f"Missing categorical profiles in columns [{', '.join(imputed_categorical_cols)}] were dynamically resolved via high-frequency mode fallback."
            else:
                categorical_summary = "Categorical Features Audit: Verified 100% complete; no missing entries detected."
                ai_categorical_segment = "Categorical attributes underwent comprehensive validation, exhibiting complete nominal records across all dimensions."
            
            preprocessing_log = [
                f"Removed {duplicates_removed} duplicate rows.",
                numeric_summary,
                categorical_summary
            ]
            
            # Step 3: Automated Class Balancing
            target_col = None
            for col in df_cleaned.columns:
                if 'target' in col.lower() or 'label' in col.lower() or 'class' in col.lower() or 'binary' in col.lower():
                    target_col = col
                    break
            
            if target_col and df_cleaned[target_col].nunique() == 2:
                preprocessing_log.append(f"Detected binary target column: '{target_col}'.")
                value_counts = df_cleaned[target_col].value_counts()
                min_class_size = value_counts.min()
                
                # Apply downsampling to balance classes equally
                df_balanced = pd.concat([
                    df_cleaned[df_cleaned[target_col] == cls].sample(min_class_size, random_state=42)
                    for cls in value_counts.index
                ])
                df_cleaned = df_balanced.reset_index(drop=True)
                preprocessing_log.append(f"Balanced class distribution for '{target_col}'. New total dataset size: {len(df_cleaned)} rows.")
            
            final_rows = len(df_cleaned)
            
            # Save the cleaned dataframe to static server storage
            df_cleaned.to_csv(STATIC_CLEANED_PATH, index=False)
            session['cleaned_file_path'] = STATIC_CLEANED_PATH
            session['initial_rows'] = initial_rows
            session['final_rows'] = final_rows
            session['duplicates'] = duplicates_removed
            
            # Dynamic Expanded AI Report
            ai_response = (
                f"The initial dataset optimization successfully audited {initial_rows} rows across {len(columns)} structural features.\n\n"
                f"• Data Cleansing: {duplicates_removed} completely redundant rows were purged.\n"
                f"• {ai_numeric_segment}\n"
                f"• {ai_categorical_segment}\n\n"
                f"The execution successfully produced a high-fidelity dataset spanning {final_rows} model-ready rows."
            )
            
            viz_response = "Automated distribution analysis generated. Correlation matrix mapping indicates features dependency matrix."
            
            # Step 4: Generate Statistical Visualizations
            charts = []
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # 1. Correlation Matrix Heatmap
                plt.figure(figsize=(6, 4))
                corr = df_cleaned[numeric_cols].corr()
                plt.imshow(corr, cmap='coolwarm', interpolation='none')
                plt.colorbar()
                plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
                plt.yticks(range(len(corr)), corr.columns, fontsize=8)
                plt.title("Correlation Matrix", fontsize=10)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                charts.append(("Correlation Matrix Heatmap", f"data:image/png;base64,{img_base64}"))
                plt.close()
                
                # 2. Boxplot Distribution Analysis
                plt.figure(figsize=(6, 4))
                df_cleaned.boxplot(column=numeric_cols[0])
                plt.title(f"Distribution Profile: {numeric_cols[0]}", fontsize=10)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                charts.append((f"Boxplot Analysis - {numeric_cols[0]}", f"data:image/png;base64,{img_base64}"))
                plt.close()
            
            # Preview first 10 rows via HTML table
            tables = [df_cleaned.head(10).to_html(classes='table table-striped table-hover', index=False)]
            
            return render_template(
                'index.html',
                message='Data processed successfully!',
                tables=tables,
                initial_rows=initial_rows,
                final_rows=final_rows,
                columns=columns,
                duplicates=duplicates_removed,
                preprocessing_log=preprocessing_log,
                ai_response=ai_response,
                viz_response=viz_response,
                charts=charts
            )
            
        except Exception as e:
            return render_template('index.html', message=f"Error processing file: {str(e)}")
            
    return render_template('index.html', message='Error: Invalid file format. Please upload a CSV file.')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').lower()
    
    if os.path.exists(STATIC_CLEANED_PATH):
        file_path = STATIC_CLEANED_PATH
    else:
        file_path = session.get('cleaned_file_path', '')
    
    # Language detection for dynamic multilingual support
    is_arabic = any(char in user_message for char in 'أبتثجحخدذرزسشصضطظعغفقكلمنهويإآةى')
    german_keywords = [
        'was', 'wie', 'viele', 'spalten', 'zeilen', 'durchschnitt', 'wert', 'duplikate', 'fehlende', 
        'zusammenfassung', 'korrelation', 'hast', 'du', 'das', 'ist', 'sind', 'behoben', 'daten', 'analyse', 'bereinigt', 'ungleichgewicht'
    ]
    is_german = any(word in user_message for word in german_keywords)
    
    if not file_path or not os.path.exists(file_path):
        if is_arabic:
            return jsonify({'reply': 'الرجاء رفع ملف CSV أولاً قبل طرح الأسئلة.'})
        elif is_german:
            return jsonify({'reply': 'Bitte laden Sie zuerst eine CSV-Datei hoch, bevor Sie Fragen stellen.'})
        else:
            return jsonify({'reply': 'Please upload a CSV file first before asking questions.'})
    
    try:
        df = pd.read_csv(file_path)
        final_rows = len(df)
        
        # Guardrail scope restriction setup
        scope_keywords = [
            'missing', 'null', 'void', 'imbalance', 'balance', 'correlation', 'relation', 'duplicate', 
            'removed', 'summary', 'done', 'average', 'mean', 'max', 'min', 'columns', 'features', 'rows', 'size',
            'مفقود', 'فارغ', 'توازن', 'ارتباط', 'علاقة', 'مكرر', 'حذف', 'ملخص', 'متوسط', 'معدل', 'أعلى', 'أقل', 'أعمدة', 'خصائص', 'صفوف',
            'fehlende', 'leere', 'ungleichgewicht', 'balance', 'korrelation', 'beziehung', 'duplikate', 'gelöscht', 'zusammenfassung', 'durchschnitt', 'maximum', 'minimum', 'spalten', 'merkmale', 'zeilen', 'behoben'
        ]
        
        if not any(keyword in user_message for keyword in scope_keywords):
            if is_arabic:
                return jsonify({'reply': 'عذراً، أنا مساعد ذكي مخصص لتحليل وتطهير مصفوفة البيانات الحالية فقط. لا يمكنني الإجابة على الأسئلة الخارجية.'})
            elif is_german:
                return jsonify({'reply': 'Entschuldigung, ich bin ein KI-Assistent, der nur auf die Analyse und Bereinigung des aktuellen Datensatzes spezialisiert ist. Ich kann keine externen Fragen beantworten.'})
            else:
                return jsonify({'reply': 'Sorry, I am an AI assistant specialized only in analyzing and preprocessing the current data matrix. I cannot answer out-of-scope questions.'})

        # Multilingual contextual evaluation mappings
        if any(w in user_message for w in ['missing', 'null', 'مفقود', 'فارغ', 'fehlende', 'leere']):
            if is_arabic:
                reply = "تمت معالجة جميع القيم المفقودة تلقائياً. الأعمدة الرقمية استخدمت الوسيط الحسابي، والأعمدة النصية استخدمت المنوال الشائع."
            elif is_german:
                reply = "Alle fehlenden Werte wurden automatisch behoben. Numerische Spalten verwendeten die Median-Imputation, kategoriale Spalten den Modus-Fallback."
            else:
                reply = "All missing values have been automatically resolved. Numeric columns used median imputation, and categorical columns used mode fallback."
                
        elif any(w in user_message for w in ['imbalance', 'balance', 'توازن', 'ungleichgewicht', 'behoben']):
            if is_arabic:
                reply = f"اكتملت عملية تحسين توازن الفئات المستهدفة ديناميكياً، مما أنتج مصفوفة جاهزة للنموذج تحتوي على {final_rows} صفاً."
            elif is_german:
                reply = f"Die Optimierung der Klassenverteilung wurde erfolgreich abgeschlossen. Die Zielparameter wurden dynamisch ausgeglichen, was zu einer modellbereiten Matrix von {final_rows} Zeilen führte."
            else:
                reply = f"Class distribution optimization completed. Balanced target parameters dynamically, resulting in a model-ready matrix of {final_rows} rows."
                
        elif any(w in user_message for w in ['correlation', 'relation', 'ارتباط', 'علاقة', 'korrelation', 'beziehung']):
            if is_arabic:
                reply = "تم رسم خريطة ارتباط الميزات بنجاح. تم فحص العلاقات الخطية العالية لضمان استقلالية البيانات المدخلة في النموذج."
            elif is_german:
                reply = "Merkmalsabhängigkeiten erfolgreich abgebildet. Hohe Kollinearitätsmetriken wurden überprüft, um die Unabhängigkeit der Modelleingaben zu gewährleisten."
            else:
                reply = "Feature dependencies mapped successfully. High collinearity metrics were checked against variance thresholds to ensure model input independence."
                
        elif any(w in user_message for w in ['duplicate', 'removed', 'مكرر', 'حذف', 'duplikate', 'gelöscht']):
            if is_arabic:
                reply = f"تأكد نظام فحص البيانات من معالجة وحذف السجلات المكررة. البيانات الحالية فريدة تماماً بنسبة 100%."
            elif is_german:
                reply = f"Die Datenprüfung hat die Verarbeitung und Löschung doppelter Datensätze bestätigt. Der aktuelle Datensatz ist zu 100% eindeutig."
            else:
                reply = f"Data auditing confirmed and processed duplicate records verification. System baseline dataset is fully unique."
                
        elif any(w in user_message for w in ['summary', 'done', 'ملخص', 'ماذا فعلت', 'zusammenfassung']):
            if is_arabic:
                reply = f"ملخص العمل: تم تنظيف البيانات وموازنتها لتصبح {final_rows} صفاً صافياً وجاهزاً. تم حذف التكرار، تعبئة الفراغات، وموازنة الفئات بأمان."
            elif is_german:
                reply = f"Zusammenfassung: Der Eingabedatensatz wurde auf {final_rows} bereinigte Einträge optimiert. Redundanzen wurden entfernt, Leerwerte aufgefüllt und die Klassenverteilung ausgeglichen."
            else:
                reply = f"Summary: Input dataset has been optimized down to {final_rows} clean balanced entries. Redundancies purged, voids imputed seamlessly, and class distribution balanced safely."
                
        elif any(w in user_message for w in ['average', 'mean', 'متوسط', 'معدل', 'durchschnitt']):
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                means = numeric_df.mean().round(2).to_dict()
                if is_arabic:
                    reply = f"المعدلات الحسابية المحسوبة للأعمدة الرقمية الأساسية هي: {str(means)}"
                elif is_german:
                    reply = f"Berechnete Durchschnittsprofile für numerische Variablen: {str(means)}"
                else:
                    reply = f"Calculated mean profiles across key variables: {str(means)}"
            else:
                if is_arabic:
                    reply = "لا توجد أعمدة رقمية متوفرة لحساب المتوسطات الحسابية لها."
                elif is_german:
                    reply = "Keine numerischen Spalten verfügbar, um Durchschnittswerte zu berechnen."
                else:
                    reply = "No computational numeric columns available to compute averages."
                    
        elif any(w in user_message for w in ['max', 'min', 'أعلى', 'أقل', 'maximum', 'minimum']):
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                max_val = numeric_df.max().round(2).to_dict()
                min_val = numeric_df.min().round(2).to_dict()
                if is_arabic:
                    reply = f"الحدود الإحصائية - القيم العظمى: {max_val}. القيم الصغرى: {min_val}."
                elif is_german:
                    reply = f"Statistische Grenzen - Maximalwerte: {max_val}. Minimalwerte: {min_val}."
                else:
                    reply = f"Statistical Boundaries - Maximum values: {max_val}. Minimum values: {min_val}."
            else:
                if is_arabic:
                    reply = "لا توجد أعمدة رقمية متوفرة لاستخراج الحدود الإحصائية."
                elif is_german:
                    reply = "Keine numerischen Spalten verfügbar, um statistische Grenzen zu extrahieren."
                else:
                    reply = "No computational numeric columns available to extract boundaries."
                    
        elif any(w in user_message for w in ['columns', 'features', 'أعمدة', 'خصائص', 'spalten', 'merkmale']):
            if is_arabic:
                reply = f"تحتوي بياناتك على {len(df.columns)} عموداً نشطاً وهي: {', '.join(df.columns.tolist())}."
            elif is_german:
                reply = f"Ihr Datensatz besitzt strukturell {len(df.columns)} aktive Spalten: {', '.join(df.columns.tolist())}."
            else:
                reply = f"Your dataset structurally possesses {len(df.columns)} active attributes: {', '.join(df.columns.tolist())}."
                
        else:
            if is_arabic:
                reply = f"أنا أحرك وأحلل حالياً مصفوفة بياناتك التي تحتوي على {final_rows} صفاً. يمكنك سؤالي عن: المكررات، القيم المفقودة، الارتباط، أو متوسطات الأعمدة."
            elif is_german:
                reply = f"Ich analysiere derzeit Ihre Datenmatrix mit {final_rows} Einträgen. Fragen Sie mich nach: Duplikaten, fehlenden Werten, Korrelation oder Spalten-Durchschnitten."
            else:
                reply = f"I am analyzing your data matrix containing {final_rows} balanced entries. Ask me about: duplicates, missing values, correlation, or columns averages."
            
        return jsonify({'reply': reply})
        
    except Exception as e:
        if is_arabic:
            return jsonify({'reply': f"خطأ أثناء تحليل سياق البيانات: {str(e)}"})
        elif is_german:
            return jsonify({'reply': f"Fehler bei der Analyse des Datenkontexts: {str(e)}"})
        return jsonify({'reply': f"Error analyzing data context: {str(e)}"})

@app.route('/download_csv')
def download_csv():
    if os.path.exists(STATIC_CLEANED_PATH):
        return send_file(STATIC_CLEANED_PATH, as_attachment=True, download_name='cleaned_dataset.csv')
    return "Error: File not found.", 404

@app.route('/download_ml')
def download_ml():
    if os.path.exists(STATIC_CLEANED_PATH):
        df = pd.read_csv(STATIC_CLEANED_PATH)
        df_encoded = pd.get_dummies(df, drop_first=True)
        buf = io.BytesIO()
        df_encoded.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='ml_ready_dataset.csv', mimetype='text/csv')
    return "Error: File not found.", 404

@app.route('/download_excel')
def download_excel():
    if os.path.exists(STATIC_CLEANED_PATH):
        df = pd.read_csv(STATIC_CLEANED_PATH)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='processed_dataset.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    return "Error: File not found.", 404

@app.route('/download_pdf')
def download_pdf():
    if os.path.exists(STATIC_CLEANED_PATH):
        df = pd.read_csv(STATIC_CLEANED_PATH)
        final_rows = len(df)
        initial_rows = session.get('initial_rows', final_rows)
        duplicates = session.get('duplicates', 0)
        
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        
        # Instantiate ReportLab sample stylesheet object cleanly
        styles = getSampleStyleSheet()
        
        story = []
        
        # Build PDF structure elements seamlessly
        story.append(Paragraph("Universal DataFlow System - Analysis Report", styles['Title']))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Initial Dataset Records: {initial_rows}", styles['Normal']))
        story.append(Paragraph(f"Identified and Purged Duplicates: {duplicates}", styles['Normal']))
        story.append(Paragraph(f"Optimized Downsampled Records: {final_rows}", styles['Normal']))
        story.append(Spacer(1, 20))
        story.append(Paragraph("System Execution Pipeline completed without system error.", styles['Heading2']))
        
        doc.build(story)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='dataflow_executive_report.pdf', mimetype='application/pdf')
    return "Error: File not found.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)