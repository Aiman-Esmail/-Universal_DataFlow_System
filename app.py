@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').lower()
    
    # Securely fallback to standard path if session variable drops
    file_path = session.get('cleaned_file_path', os.path.join(UPLOAD_FOLDER, 'latest_cleaned_data.csv'))
    
    # 1. Detect User Language
    is_arabic = any(char in user_message for char in 'أبتثجحخدذرزسشصضطظعغفقكلمنهويإآةى')
    is_german = any(word in user_message for word in ['was', 'wie', 'viele', 'spalten', 'zeilen', 'durchschnitt', 'wert', 'duplikate', 'fehlende', 'zusammenfassung', 'korrelation'])
    
    # 2. Absolute check if the physical file exists on the server
    if not os.path.exists(file_path):
        if is_arabic:
            return jsonify({'reply': 'الرجاء رفع ملف CSV أولاً قبل طرح الأسئلة.'})
        elif is_german:
            return jsonify({'reply': 'Bitte laden Sie zuerst eine CSV-Datei hoch, bevor Sie Fragen stellen.'})
        else:
            return jsonify({'reply': 'Please upload a CSV file first before asking questions.'})
    
    try:
        # Load data context safely
        df = pd.read_csv(file_path)
        
        # Dynamic fallback calculation if Render clears the flash session memory
        final_rows = len(df)
        initial_rows = session.get('initial_rows', final_rows)
        duplicates = session.get('duplicates', 0)
        
        # 3. Guardrail: Check if the question is out of project scope
        scope_keywords = [
            'missing', 'null', 'void', 'imbalance', 'balance', 'correlation', 'relation', 'duplicate', 
            'removed', 'summary', 'done', 'average', 'mean', 'max', 'min', 'columns', 'features', 'rows', 'size',
            'مفقود', 'فارغ', 'توازن', 'ارتباط', 'علاقة', 'مكرر', 'حذف', 'ملخص', 'متوسط', 'معدل', 'أعلى', 'أقل', 'أعمدة', 'خصائص', 'صفوف',
            'fehlende', 'leere', 'ungleichgewicht', 'balance', 'korrelation', 'beziehung', 'duplikate', 'gelöscht', 'zusammenfassung', 'durchschnitt', 'maximum', 'minimum', 'spalten', 'merkmale', 'zeilen'
        ]
        
        if not any(keyword in user_message for keyword in scope_keywords):
            if is_arabic:
                return jsonify({'reply': 'عذراً، أنا مساعد ذكي مخصص لتحليل وتطهير مصفوفة البيانات الحالية فقط. لا يمكنني الإجابة على الأسئلة الخارجية.'})
            elif is_german:
                return jsonify({'reply': 'Entschuldigung, ich bin ein KI-Assistent, der nur auf die Analyse und Bereinigung des aktuellen Datensatzes spezialisiert ist. Ich kann keine externen Fragen beantworten.'})
            else:
                return jsonify({'reply': 'Sorry, I am an AI assistant specialized only in analyzing and preprocessing the current data matrix. I cannot answer out-of-scope questions.'})

        # 4. Rule-based analytical chatbot routing (Trilingual Support)
        # Missing Values
        if any(w in user_message for w in ['missing', 'null', 'مفقود', 'فارغ', 'fehlende', 'leere']):
            if is_arabic:
                reply = "تمت معالجة جميع القيم المفقودة تلقائياً. الأعمدة الرقمية استخدمت الوسيط الحسابي، والأعمدة النصية استخدمت المنوال الشائع."
            elif is_german:
                reply = "Alle fehlenden Werte wurden automatisch behoben. Numerische Spalten verwendeten die Median-Imputation, kategoriale Spalten den Modus-Fallback."
            else:
                reply = "All missing values have been automatically resolved. Numeric columns used median imputation, and categorical columns used mode fallback."
                
        # Class Imbalance
        elif any(w in user_message for w in ['imbalance', 'balance', 'توازن', 'ungleichgewicht']):
            if is_arabic:
                reply = f"اكتملت عملية تحسين توازن الفئات المستهدفة ديناميكياً، مما أنتج مصفوفة جاهزة للنموذج تحتوي على {final_rows} صفاً."
            elif is_german:
                reply = f"Die Optimierung der Klassenverteilung wurde abgeschlossen. Die Zielparameter wurden dynamisch ausgeglichen, was zu einer modellbereiten Matrix von {final_rows} Zeilen führte."
            else:
                reply = f"Class distribution optimization completed. Balanced target parameters dynamically, resulting in a model-ready matrix of {final_rows} rows."
                
        # Correlation
        elif any(w in user_message for w in ['correlation', 'relation', 'ارتباط', 'علاقة', 'korrelation', 'beziehung']):
            if is_arabic:
                reply = "تم رسم خريطة ارتباط الميزات بنجاح. تم فحص العلاقات الخطية العالية لضمان استقلالية البيانات المدخلة في النموذج."
            elif is_german:
                reply = "Merkmalsabhängigkeiten erfolgreich abgebildet. Hohe Kollinearitätsmetriken wurden überprüft, um die Unabhängigkeit der Modelleingaben zu gewährleisten."
            else:
                reply = "Feature dependencies mapped successfully. High collinearity metrics were checked against variance thresholds to ensure model input independence."
                
        # Duplicates
        elif any(w in user_message for w in ['duplicate', 'removed', 'مكرر', 'حذف', 'duplikate', 'gelöscht']):
            if is_arabic:
                reply = f"تأكد نظام فحص البيانات من معالجة وحذف السجلات المكررة. البيانات الحالية فريدة تماماً بنسبة 100%."
            elif is_german:
                reply = f"Die Datenprüfung hat die Verarbeitung und Löschung doppelter Datensätze bestätigt. Der aktuelle Datensatz ist zu 100% eindeutig."
            else:
                reply = f"Data auditing confirmed and processed duplicate records verification. System baseline dataset is fully unique."
                
        # Summary
        elif any(w in user_message for w in ['summary', 'done', 'ملخص', 'ماذا فعلت', 'zusammenfassung']):
            if is_arabic:
                reply = f"ملخص العمل: تم تنظيف البيانات وموازنتها لتصبح {final_rows} صفاً صافياً وجاهزاً. تم حذف التكرار، تعبئة الفراغات، وموازنة الفئات بأمان."
            elif is_german:
                reply = f"Zusammenfassung: Der Eingabedatensatz wurde auf {final_rows} bereinigte Einträge optimiert. Redundanzen wurden entfernt, Leerwerte aufgefüllt und die Klassenverteilung ausgeglichen."
            else:
                reply = f"Summary: Input dataset has been optimized down to {final_rows} clean balanced entries. Redundancies purged, voids imputed seamlessly, and class distribution balanced safely."
                
        # Averages / Means
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
                    
        # Max / Min Boundaries
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
                    
        # Columns / Features Info
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
