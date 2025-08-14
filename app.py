# app.py
from flask import Flask, request, jsonify
import joblib
import string
import nltk

# Descargar stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Cargar modelo y vectorizador desde carpeta principal
modelo = joblib.load("modelo.pkl")
vectorizador = joblib.load("vectorizador.pkl")

# Función para limpiar texto
stop_words = set(stopwords.words('spanish'))
def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = [p for p in texto.split() if p not in stop_words]
    return ' '.join(palabras)

# Crear app Flask
app = Flask(__name__)

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    texto = data.get("texto", "")
    texto_limpio = limpiar_texto(texto)
    texto_vectorizado = vectorizador.transform([texto_limpio])
    prediccion = modelo.predict(texto_vectorizado)[0]
    return jsonify({"prediccion": prediccion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
