from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modeloRf93.pkl')
scaler = joblib.load('dataFrameScalado.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        ram = float(request.form['ram'])
        peso = float(request.form['peso'])
        ppi = float(request.form['ppi'])
        cpu = float(request.form['cpu'])
        discoDuro = float(request.form['discoDuro'])
        soWindows = float(request.form['soWindows'])

        input_data = pd.DataFrame({
            'Ram': [ram],
            'Weight': [peso],
            'TouchScreen': [0],
            'Ips': [0],
            'Ppi': [ppi],
            'Cpu_brand': [cpu],
            'HDD': [0],
            'SSD': [discoDuro],
            'Os_Mac': [0],
            'Os_Others': [0],
            'Os_Windows': [soWindows],
            'Gpu_brand_AMD': [0],
            'Gpu_brand_Intel': [0],
            'Gpu_brand_Nvidia': [0]
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [0, 1, 4, 5, 7, 10]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = round(float(prediccion[0]), 2)

        return jsonify({'prediction': prediction_value})
     
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
