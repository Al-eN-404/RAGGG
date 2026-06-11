import os
import weaviate
import weaviate.classes as wvc
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def get_weaviate_client():
    """
    Establish a connection to the Weaviate v4 instance.
    Supports local Docker setup or Weaviate Cloud (WCD) cluster configuration.
    """
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080").strip()
    api_key = os.getenv("WEAVIATE_API_KEY")
    
    # Auto-prepend scheme if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        if "localhost" in url or "127.0.0.1" in url:
            url = "http://" + url
        else:
            url = "https://" + url
            
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    
    # Check if the endpoint points to Weaviate Cloud (WCD)
    is_cloud = host.endswith(".weaviate.network") or host.endswith(".weaviate.cloud")
    auth = wvc.init.Auth.api_key(api_key) if api_key else None
    
    if is_cloud:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=auth
        )
    else:
        # Default to local or custom url
        if host in ("localhost", "127.0.0.1"):
            return weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=50051,
                auth_credentials=auth
            )
        else:
            return weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                grpc_host=host,
                grpc_port=50051,
                http_secure=parsed.scheme == "https",
                grpc_secure=parsed.scheme == "https",
                auth_credentials=auth
            )

def build_vectorstore(chunks):
    """
    Encode chunks locally using SentenceTransformer and upload them to Weaviate in batch.
    """
    # Initialize the local embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Extract texts and compute embeddings locally
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    with get_weaviate_client() as client:
        collection_name = "WebsiteData"
        
        # Delete if exists to rebuild fresh
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            
        # Create collection without server-side vectorizer (we supply vectors)
        collection = client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
            ],
            vectorizer_config=None
        )
        
        # Batch upload to Weaviate
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                vector = embeddings[i].tolist()  # Convert numpy array to list of floats
                batch.add_object(
                    properties={
                        "text": chunk["text"],
                        "url": chunk["url"]
                    },
                    vector=vector
                )
        print(f"Ingested {len(chunks)} chunks into Weaviate.")