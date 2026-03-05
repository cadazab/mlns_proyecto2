import streamlit as st
import joblib
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess(text):
    tokenizer = RegexpTokenizer(r"\w+")
    stemmer = PorterStemmer()
    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("spanish")]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

@st.cache_resource
def load_model():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        model = joblib.load("best_model_pipeline_final.pkl")
        return model
    except FileNotFoundError:
        st.error(f"⚠️ Error: No se encontró el archivo 'best_model_pipeline_final.pkl'")
        return None
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {str(e)}")
        return None

model = load_model()

st.title('Clasificador de Textos para ODS')
st.text('Proyecto Desarrollado por: [Camilo Andres Daza Barrios y Neill Rolando Giraldo Corredor]')
st.write('Introduce un texto para clasificarlo según los Objetivos de Desarrollo Sostenible (ODS).')

# tranform this object  into list of tuples  
ods_tuples = [
    ("Fin de la pobreza", "Poner fin a la pobreza en todas sus formas en todo el mundo."),
    ("Hambre cero", "Poner fin al hambre, lograr la seguridad alimentaria y la mejora de la nutrición y promover la agricultura sostenible."),
    ("Salud y bienestar", "Garantizar una vida sana y promover el bienestar para todos en todas las edades."),
    ("Educación de calidad", "Garantizar una educación inclusiva, equitativa y de calidad y promover oportunidades de aprendizaje durante toda la vida para todos."),
    ("Igualdad de género", "Lograr la igualdad entre los géneros y empoderar a todas las mujeres y las niñas."),
    ("Agua limpia y saneamiento", "Garantizar la disponibilidad de agua y su gestión sostenible y el saneamiento para todos."),
    ("Energía asequible y no contaminante", "Garantizar el acceso a una energía asequible, segura, sostenible y moderna para todos."),
    ("Trabajo decente y crecimiento económico", "Promover el crecimiento económico sostenido, inclusivo y sostenible, el empleo pleno y productivo y el trabajo decente para todos."),
    ("Industria, innovación e infraestructura", "Construir infraestructuras resilientes, promover la industrialización inclusiva y sostenible y fomentar la innovación."),
    ("Reducción de las desigualdades", "Reducir la desigualdad en y entre los países."),
    ("Ciudades y comunidades sostenibles", "Lograr que las ciudades y los asentamientos humanos sean inclusivos, seguros, resilientes y sostenibles."),
    ("Producción y consumo responsables", "Garantizar modalidades de consumo y producción sostenibles."),
    ("Acción por el clima", "Adoptar medidas urgentes para combatir el cambio climático y sus efectos."),
    ("Vida submarina", "Conservar y utilizar en forma sostenible los océanos, los mares y los recursos marinos para el desarrollo sostenible."),
    ("Vida de ecosistemas terrestres", "Gestionar sosteniblemente los bosques, luchar contra la desertificación, detener e invertir la degradación de las tierras y detener la pérdida de biodiversidad."),
    ("Paz, justicia e instituciones sólidas", "Promover sociedades, justas, pacíficas e inclusivas."),
    ("Alianzas para lograr los objetivos", "Fortalecer los medios de ejecución y revitalizar la Alianza Mundial para el Desarrollo Sostenible.")
]     
        
with st.sidebar:
    st.expander("", expanded=True)
    st.write("Objetivos de Desarrollo Sostenible (ODS):")
    index = 1
    for ods, descripcion in ods_tuples:
        st.markdown(f"**{index}. {ods}**: {descripcion}")    
        index += 1
        
if model is None:
    st.stop()

user_input = st.text_area('Ingresa tu texto aquí:', '')

if st.button('Clasificar'):
    if user_input:
        # Predecir el ODS - pasar como lista de strings
        prediction = model.predict([user_input])
        st.success(f'El texto se clasifica como ODS: {prediction[0]}: {ods_tuples[prediction[0]-1][0]}')
    else:
        st.warning('Por favor, ingresa un texto para clasificar.')