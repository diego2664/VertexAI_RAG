import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from get_embedding_function_vertex import get_embedding_function
from google.oauth2 import service_account

# Definir la ruta de la carpeta de datos
DATA_PATH = "data"

# Cargar los documentos PDF
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

# Obtener la función de embedding
embedding_function = get_embedding_function()

# Procesar cada documento
for doc in documents:
    # Obtener el contenido del documento
    content = doc.page_content
    
    # Generar el embedding
    embedding = embedding_function.embed_query(content)
    
    # Imprimir información sobre el documento y su embedding
    print(f"Documento: {doc.metadata['source']}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    print(f"Dimensiones del vector: {len(embedding)}")
    print(f"Primeros 5 valores del vector: {embedding[:5]}")
    print("-" * 50)

    # Importar las bibliotecas necesarias de Google Cloud
    from google.cloud import aiplatform
    from google.cloud.aiplatform import MatchingEngineIndex
    from dotenv import load_dotenv

    # Cargar variables de entorno
    load_dotenv()

    # Configurar el proyecto y la ubicación
    PROJECT_ID = os.getenv('GCP_PROJECT_NAME')
    LOCATION = os.getenv('VERTEX_AI_LOCATION')
    INDEX_NAME = os.getenv('INDEX_NAME')
    credentials = service_account.Credentials.from_service_account_file("./sp-controlfinanza-plygrd-4e1f58185d71-service-account.json")

    print("credenciales = ",credentials)

    # Inicializar el cliente de Vertex AI con la cuenta de servicio
    aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

    # Intentar obtener el índice existente o crear uno nuevo
    try:
        index = MatchingEngineIndex.(index_name=INDEX_NAME)
        print(f"Índice '{INDEX_NAME}' encontrado.")
    except Exception:
        print(f"Creando nuevo índice '{INDEX_NAME}'...")
        index = MatchingEngineIndex.create(
            display_name=INDEX_NAME,
            dimensions=len(embedding),
            metadata_config={
                "id": "STRING",
                "source": "STRING",
                "page": "INT64",
            }
        )
        print(f"Índice '{INDEX_NAME}' creado exitosamente.")

    # Preparar datos para upsert
    embeddings = [embedding]
    ids = [f"{doc.metadata['source']}:{doc.metadata.get('page', 'N/A')}"]
    metadatas = [{
        "source": doc.metadata['source'],
        "page": doc.metadata.get('page', 'N/A'),
    }]

    # Realizar upsert de vectores al índice
    index_endpoint = index.deploy()
    matching_engine_index_endpoint = index_endpoint.gca_resource

    matching_engine_index_endpoint.upsert_embeddings(
        embeddings=embeddings,
        ids=ids,
        deployment_name=index.deployed_index.id
    )

    print(f"✅ Vector añadido/actualizado en el índice '{INDEX_NAME}' de Vertex AI")
