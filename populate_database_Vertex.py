import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function_vertex import get_embedding_function
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"
PROJECT_ID = os.getenv('GCP_PROJECT_NAME')
LOCATION = os.getenv('VERTEX_AI_LOCATION')
INDEX_ID = os.getenv('VERTEX_AI_INDEX_ID')


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_vertex(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

def add_to_vertex(chunks: list[Document]):
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Obtener o crear el índice
    try:
        index = MatchingEngineIndex(index_name=INDEX_ID)
    except Exception:
        # Si el índice no existe, créalo
        index = MatchingEngineIndex.create(
            display_name=INDEX_ID,
            dimensions=768,  # Ajusta esto según la dimensión de tus embeddings
            metadata_config={
                "id": "STRING",
                "source": "STRING",
                "page": "INT64",
            }
        )

    # Calcular IDs de chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Obtener embeddings
    embedding_function = get_embedding_function()
    
    # Preparar datos para upsert
    embeddings = []
    ids = []
    metadatas = []

    for chunk in chunks_with_ids:
        embedding = embedding_function.embed_query(chunk.page_content)
        embeddings.append(embedding)
        ids.append(chunk.metadata["id"])
        metadatas.append({
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
        })

    # Upsert de vectores al índice
    index_endpoint = index.deploy()
    matching_engine_index_endpoint = index_endpoint.gca_resource

    matching_engine_index_endpoint.upsert_embeddings(
        embeddings=embeddings,
        ids=ids,
        deployment_name=index.deployed_index.id
    )

    print(f"✅ Añadidos/actualizados {len(chunks)} documentos en el índice de Vertex AI")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
