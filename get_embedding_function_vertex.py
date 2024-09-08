from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding_function():
    # Modelo: text-embedding-gecko
    # Dimensiones del vector: 768
    embeddings = VertexAIEmbeddings(
        model_name="text-multilingual-embedding-002",
        project=os.getenv('GCP_PROJECT_NAME'),
        location=os.getenv('VERTEX_AI_LOCATION')
    )
    return embeddings

# Asegúrate de inicializar la aplicación de Google Cloud antes de usar esta función
# aiplatform.init(project="tu-proyecto-id", location="us-central1")