import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Descargar stopwords de NLTK (solo la primera vez)
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- LIMPIEZA DE TEXTO ---
stop_words = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words]
    return ' '.join(palabras)

# --- CARGA Y ENTRENAMIENTO DEL MODELO ---
df = pd.read_csv("dataset.csv")  # Asegúrate de que el archivo esté en la misma carpeta

df['texto_limpio'] = df['texto'].apply(limpiar_texto)

vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(df['texto_limpio'])
y = df['etiqueta']

modelo = MultinomialNB()
modelo.fit(X, y)

# --- FUNCIÓN DE PREDICCIÓN ---
def clasificar_texto(texto_usuario):
    texto_limpio = limpiar_texto(texto_usuario)
    texto_vectorizado = vectorizador.transform([texto_limpio])
    prediccion = modelo.predict(texto_vectorizado)[0]
    return prediccion
from flask import Flask, render_template, request
from modelo import clasificar_texto

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        texto_usuario = request.form["texto"]
        resultado = clasificar_texto(texto_usuario)
    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)

