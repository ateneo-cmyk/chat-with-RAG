import cohere
import streamlit as st
import numpy as np
import PyPDF2

# Configuración de la interfaz
st.set_page_config(page_title="Mi RAG con Cohere", layout="wide")
st.title("📚 Chat con mis Documentos (RAG)")


#subir pdfs para el sistema rag
with st.sidebar:
    st.header("sube un archivo")
    archivo_subido = st.file_uploader("Sube un archivo PDF", type ="pdf")
    
    documentos_cohere = []
    
    if archivo_subido is not None:
        with st.spinner("Leyendo pdf..."):
           #Leer el PDF
           lector_pdf = PyPDF2.PdfReader(archivo_subido)
           texto_completo = ""
           
           for pagina in lector_pdf.pages:
               texto_completo +=pagina.extract_text() + "\n"
           
        #Dividir el texto en fragmentos (chunks) para cohere
        fragmentos = texto_completo.split('\n\n')
        
        for i, fragmentos in enumerate(fragmentos):
          if len(fragmentos.strip()) > 20:
              documentos_cohere.append({
                  "id": f"frag_{i}",
                  "text": fragmentos.strip()
              })
        st.success(f"¡PDF procesado! Se extrajeron {len(documentos_cohere)} fragmentos.")
# --- CONFIGURACIÓN DE SEGURIDAD ---
#obtener la clave desde secrets de Streamlit
try:
    api_key = st.secrets["COHERE_API_KEY"]
    client = cohere.Client(api_key)
except Exception:
    st.error("⚠️ Falta la API Key de Cohere. Agrégala en .streamlit/secrets.toml o en la web de Streamlit.")
    st.stop()

# --- MEMORIA DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INTERACCIÓN ---
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    #Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #Generar respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                #historial en el formato exacto que pide la API de Cohere
                history = [
                    {"role": m["role"].upper() if m["role"] != "assistant" else "CHATBOT", "message": m["content"]} 
                    for m in st.session_state.messages[:-1]
                ]
                
                response = client.chat(
                    message=prompt,
                    chat_history=history,
                    model="command-a-03-2025",
                    documents=documentos_cohere if documentos_cohere else None 
                )
                
                answer = response.text
                st.markdown(answer)
                
                #Guardar solo el texto de la respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error en la comunicación con la IA: {e}")
                
                
