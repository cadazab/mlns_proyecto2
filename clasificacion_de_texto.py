import streamlit as st
import joblib
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descargar stopwords si no están ya descargadas (solo se ejecuta una vez)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Definir la función de preprocesamiento (la misma que usaste en el notebook)
def preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('spanish')]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Error: No se encontró el archivo 'best_model_pipeline.pkl'")
        return None
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {str(e)}")
        return None

model = load_model()

st.title('Clasificador de Textos para ODS')
st.write('Introduce un texto para clasificarlo según los Objetivos de Desarrollo Sostenible (ODS).')

if model is None:
    st.stop()

user_input = st.text_area('Ingresa tu texto aquí:', '')

if st.button('Clasificar'):
    if user_input:
        # Predecir el ODS
        prediction = model.predict([user_input])
        st.success(f'El texto se clasifica como ODS: {prediction[0]}')
    else:
        st.warning('Por favor, ingresa un texto para clasificar.')