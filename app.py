from flask import Flask, request, jsonify, render_template
import pandas as pd
from flask_docs import ApiDoc

# Вся ML model.py
from model import string_analyse, tests_analyse

app = Flask(__name__, static_folder='static', template_folder='templates')

ApiDoc(
    app,
    title="AIJIC",
    version="1.0.0",
    description="AIJIC hackathon app",
)

# API для анализа текста, возвращает JSON
@app.route('/analyse', methods=['POST'])
def analyse_text():
    
    try:
        data = request.json
        text = data.get("text", "")
        if text:
            result = string_analyse(text)
            return jsonify({"status": "success", "result": result}), 200
        else:
            return jsonify({"status": "error", "message": "No text provided"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API для загрузки CSV и анализа, возвращает JSON
@app.route('/upload', methods=['POST'])
def upload_csv():
    try:
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            test_data = pd.read_csv(file)
            result = tests_analyse(test_data)
            return result.to_json(orient="records"), 200
        else:
            return jsonify({"status": "error", "message": "Invalid file format"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Маршрут для отображения фронтенда
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port="5500", debug=True)
