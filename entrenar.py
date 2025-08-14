# entrenar.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import string
import joblib

# Descargar stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Leer dataset (asegúrate de tener dataset.csv en el repo)
df = pd.read_csv("dataset.csv")

# Limpiar texto
stop_words = set(stopwords.words('spanish'))
def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = [p for p in texto.split() if p not in stop_words]
    return ' '.join(palabras)

df['texto_limpio'] = df['texto'].apply(limpiar_texto)

# Vectorizar
vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(df['texto_limpio'])
y = df['etiqueta']

# Entrenar
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = MultinomialNB()
modelo.fit(X_entrenamiento, y_entrenamiento)

# Guardar modelo y vectorizador
joblib.dump(modelo, "modelo.pkl")
joblib.dump(vectorizador, "vectorizador.pkl")

print("✅ Modelo y vectorizador guardados correctamente.")
