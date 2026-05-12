from flask import Flask, render_template, request, jsonify
import pandas as pd
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded"}), 400
    
    file = request.files['file']
    # Get the question from the form
    user_question = request.form.get('question', 'No question asked.')
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=None, engine='python')
        
        # Here we simulate AI response based on the question and data
        response_message = f"File processed successfully. Regarding your question: '{user_question}', the AI is analyzing the {len(df.columns)} columns."
        
        return jsonify({
            "message": response_message,
            "rows": len(df),
            "cols": len(df.columns)
        })

if __name__ == '__main__':
    app.run(debug=True)
