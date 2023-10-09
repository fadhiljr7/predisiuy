from flask import Flask, request, render_template
import pickle
import locale
import numpy as np
import pandas as pd

app = Flask(__name__)

# Memuat model Regresi Linier dan Random Forest dari file pickle.
with open('linreg_model.pkl', 'rb') as f:
    linreg_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Membaca Data Set
dataset = pd.read_csv('DATA RUMAH.csv', sep=';')
dataset = dataset.rename(columns={'NAMA RUMAH': 'NAMA_RUMAH'})
dataset = dataset.drop(columns='NO')

# Atur locale ke bahasa Indonesia
locale.setlocale(locale.LC_ALL, 'id_ID')

# Set format mata uang sebagai Rupiah
currency_format = locale.currency

# Fungsi untuk menghitung rekomendasi rumah berdasarkan hasil prediksi
def convert_to_float(value):
    if isinstance(value, str):
        value = value.replace('Rp', '').replace('.', '').replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return np.nan

def calculate_similarity(predicted_price, actual_price):
    predicted_price = convert_to_float(predicted_price)  # Convert to float
    actual_price = convert_to_float(actual_price)  # Convert to float

    similarity = 100 - (abs(predicted_price - actual_price) / actual_price) * 100
    return similarity

def get_rekomendasi(linreg_prediction, rf_prediction, num_samples=5):
    # Convert predictions to float using the custom function
    linreg_prediction = convert_to_float(linreg_prediction)
    rf_prediction = convert_to_float(rf_prediction)

    # Calculate similarity between predicted prices and prices in the dataset
    dataset['Similarity_linreg'] = dataset.apply(
        lambda row: calculate_similarity(linreg_prediction, row['HARGA']), axis=1)
    dataset['Similarity_rf'] = dataset.apply(
        lambda row: calculate_similarity(rf_prediction, row['HARGA']), axis=1)

    # Combine the similarities and sort the dataset based on the similarity
    dataset['Similarity'] = (dataset['Similarity_linreg'] + dataset['Similarity_rf']) / 2
    dataset.sort_values(by='Similarity', ascending=False, inplace=True)

    # Get the random recommendations
    random_rekomendasi = dataset.sample(n=num_samples, random_state=42)

    return random_rekomendasi.to_dict('records')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        lb = float(request.form['lb'])
        lt = float(request.form['lt'])
        kt = int(request.form['kt'])
        km = int(request.form['km'])
        GRS = int(request.form['GRS'])

        input_data = [[lb, lt, kt, km, GRS]]
        
          # Get predictions for each model.
        linreg_prediction = linreg_model.predict(input_data)[0]
        rf_prediction = rf_model.predict(input_data)[0]

        # Ubah hasil prediksi menjadi format mata uang Rupiah
        linreg_prediction = currency_format(linreg_prediction, grouping=True)
        rf_prediction = currency_format(rf_prediction, grouping=True)
      
       # Get random recommended houses based on the input data and predicted prices
        rekomendasi = get_rekomendasi(linreg_prediction, rf_prediction, num_samples=5)

        return render_template('index.html', linreg_prediction=linreg_prediction, rf_prediction=rf_prediction,
                               rekomendasi=rekomendasi, hasil_input=request.form)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
