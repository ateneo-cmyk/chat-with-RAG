import cohere
import streamlit as st
import numpy as np
import PyPDF2
import chromadb

# Configuración de la interfaz
st.set_page_config(page_title="Mi RAG con Cohere", layout="wide")
st.title("📚 Chat con mis Documentos (RAG)")

# --- CONFIGURACIÓN DE SEGURIDAD 
try:
    api_key = st.secrets["COHERE_API_KEY"]
    client = cohere.Client(api_key)
except Exception:
    st.error("⚠️ Falta la API Key de Cohere. Agrégala en .streamlit/secrets.toml o en la web de Streamlit.")
    st.stop()

# --- BASE DE DATOS VECTORIAL ---
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()
    # Es mejor usar guión bajo en lugar de espacios para los nombres de colecciones
    st.session_state.coleccion = st.session_state.chroma_client.get_or_create_collection(name="mis_documentos")
    st.session_state.pdf_procesado = False

# --- SUBIR PDFs ---
with st.sidebar:
    st.header("Sube un archivo")
    archivo_subido = st.file_uploader("Sube un archivo PDF", type ="pdf")
    
    # Procesar solo si hay archivo Y NO ha sido procesado aún
    if archivo_subido is not None and not st.session_state.pdf_procesado:
        with st.spinner("Leyendo y procesando el PDF..."):
            lector_pdf = PyPDF2.PdfReader(archivo_subido)
            texto_completo = ""
            
            for pagina in lector_pdf.pages:
                texto_completo += pagina.extract_text() + "\n"
            
            fragmentos = texto_completo.split('\n\n')
            textos_limpios = [frag.strip() for frag in fragmentos if len(frag.strip()) > 20]
            
            if textos_limpios:
                try:
                    # Usamos el mismo modelo multilingüe para guardar
                    respuesta_embeddings = client.embed(
                        model="embed-multilingual-v3.0", 
                        texts=textos_limpios,
                        input_type="search_document", 
                    )
                    
                    # Corrección al plural
                    embeddings = respuesta_embeddings.embeddings
                    
                    # Corrección del formato de los IDs
                    ids = [f"frag_{i}" for i in range(len(textos_limpios))]
                    
                    st.session_state.coleccion.add(
                        documents=textos_limpios,
                        embeddings=embeddings,
                        ids=ids
                    )
                    
                    st.session_state.pdf_procesado = True
                    st.success(f"¡PDF guardado en la base de datos vectorial con {len(textos_limpios)} fragmentos!")
                except Exception as e:
                    st.error(f"Error al generar embeddings: {e}")

# --- MEMORIA DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INTERACCIÓN ---
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                documentos_relevantes = None
                
                # Buscamos si hay un pdf procesado
                if "pdf_procesado" in st.session_state and st.session_state.pdf_procesado:
                    
                    pregunta_vectorizada = client.embed(
                        texts=[prompt],
                        model="embed-multilingual-v3.0", # Mismo modelo que al guardar
                        input_type="search_query"
                    ).embeddings[0]
                    
                    resultados_busqueda = st.session_state.coleccion.query(
                        query_embeddings=[pregunta_vectorizada],
                        n_results=3
                    )
                    
                    documentos_relevantes = []
                    if resultados_busqueda['documents'] and len(resultados_busqueda['documents'][0]) > 0:
                        for i, texto_recuperado in enumerate(resultados_busqueda['documents'][0]):
                            documentos_relevantes.append({
                                "id": f"doc_{i}", 
                                "text": texto_recuperado
                            })
                
                history = [
                    {"role": m["role"].upper() if m["role"] != "assistant" else "CHATBOT", "message": m["content"]} 
                    for m in st.session_state.messages[:-1]
                ]
                
                response = client.chat(
                    message=prompt,
                    chat_history=history,
                    model="command-a-03-2025",
                    documents=documentos_relevantes if documentos_relevantes else None
                )
                
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")