import cohere
import streamlit as st
import numpy as np

# Configuración de la interfaz
st.set_page_config(page_title="Mi RAG con Cohere", layout="wide")
st.title("📚 Chat con mis Documentos (RAG)")

# --- CONFIGURACIÓN DE SEGURIDAD ---
# Intentamos obtener la clave desde secrets de Streamlit
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
    # 1. Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # Preparamos el historial en el formato exacto que pide la API de Cohere
                history = [
                    {"role": m["role"].upper() if m["role"] != "assistant" else "CHATBOT", "message": m["content"]} 
                    for m in st.session_state.messages[:-1]
                ]
                
                response = client.chat(
                    message=prompt,
                    chat_history=history,
                    model="command-r-plus" 
                )
                
                answer = response.text
                st.markdown(answer)
                
                # 3. Guardar solo el texto de la respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error en la comunicación con la IA: {e}")