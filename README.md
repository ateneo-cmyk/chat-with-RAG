# 📄 Chat con  Documentos (Sistema RAG)

Una aplicación web interactiva que permite a los usuarios cargar archivos PDF y realizar consultas en lenguaje natural sobre su contenido. El sistema utiliza Generación Aumentada por Recuperación (RAG) para encontrar la información más relevante y generar respuestas precisas.

## 🚀 Características Principales
*   **Carga de documentos:** Soporte para lectura y procesamiento de archivos PDF.
*   **Búsqueda semántica:** Recuperación de contexto basada en similitud de vectores.
*   **Respuestas inteligentes:** Generación de texto fluido y fundamentado en los documentos proporcionados.

## 🏗️ Arquitectura y Stack Tecnológico
El flujo de procesamiento de los documentos y consultas está diseñado siguiendo el patrón arquitectónico de **Pipes and Filters**, asegurando un procesamiento modular y escalable.

*   **Interfaz de Usuario:** Streamlit
*   **Modelo de Lenguaje (LLM):** Cohere
*   **Base de Datos Vectorial:** ChromaDB
*   **Gestión de Dependencias:** Python / Pip

## ⚙️ Instalación y Uso Local

Sigue estos pasos para correr el proyecto en tu máquina:

Crear un entorno virtual:

Bash
python -m venv venv
Activar el entorno virtual:

En Windows (Git Bash): source venv/Scripts/activate

En Windows (CMD/PowerShell): venv\\Scripts\\activate

En Linux o macOS: source venv/bin/activate

Instalar las dependencias:

Bash
pip install -r requirements.txt
Configurar Variables de Env (Secrets):
Crea un archivo .env en la raíz con:
COHERE_API_KEY=tu_clave_api_aqui

Ejecutar la aplicación:

Bash
streamlit run app.py
