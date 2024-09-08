import vertexai
import os
import sys
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account

# NOTE : This operation can take upto 30 seconds

# Inicializar la instancia de Google Cloud
load_dotenv()

DISPLAY_NAME = "testcontratos"
GCP_PROJECT_NAME = os.getenv('GCP_PROJECT_NAME')
VERTEX_AI_LOCATION = os.getenv('VERTEX_AI_LOCATION')
INDEX_NAME = os.getenv('INDEX_NAME')
# Cargar las credenciales del archivo de cuenta de servicio
credentials = service_account.Credentials.from_service_account_file('./sp-controlfinanza-plygrd-4e1f58185d71-service-account.json')
# Configurar el proyecto y la región
project_id = GCP_PROJECT_NAME
region = VERTEX_AI_LOCATION  # Por ejemplo, 'us-central1'
print(project_id)
print(region)

# Inicializar el cliente de Vertex AI
aiplatform.init(
    project=project_id,
    location=region,
    credentials=credentials
)

print(f"Instancia de Google Cloud inicializada para el proyecto {project_id} en la región {region}")

my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=DISPLAY_NAME,
    dimensions=768,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="STREAM_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE
)
print("Index created")
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=f"{DISPLAY_NAME}-endpoint", public_endpoint_enabled=True
)