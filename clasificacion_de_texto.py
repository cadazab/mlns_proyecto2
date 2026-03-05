import streamlit as st
import joblib

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
        # Predecir el ODS - pasar como lista de strings
        prediction = model.predict(user_input)
        st.success(f'El texto se clasifica como ODS: {prediction[0]}')
    else:
        st.warning('Por favor, ingresa un texto para clasificar.')