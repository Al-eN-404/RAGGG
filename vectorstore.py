import os
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

load_dotenv()

def get_secret(key_name, default=None):
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default)

def get_huggingface_embeddings(texts):
    """
    Generate embeddings using Hugging Face's serverless Inference API.
    If the API call fails or is unreachable, falls back to generating
    embeddings locally using the sentence-transformers library.
    """
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
        
    try:
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        
        hf_token = get_secret("HF_TOKEN")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "inputs": batch,
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 503:
                import time
                for _ in range(5):
                    time.sleep(3)
                    response = requests.post(api_url, headers=headers, json=payload, timeout=10)
                    if response.status_code != 503:
                        break
                        
            if response.status_code != 200:
                raise Exception(f"Hugging Face API error ({response.status_code}): {response.text}")
                
            batch_embeddings = response.json()
            if not isinstance(batch_embeddings, list):
                raise ValueError(f"Unexpected embeddings format from Hugging Face API: {type(batch_embeddings)}")
                
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings[0] if is_single else all_embeddings
        
    except Exception as api_err:
        print(f"Hugging Face Inference API failed: {api_err}. Falling back to local SentenceTransformer...")
        try:
            from sentence_transformers import SentenceTransformer
            local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            local_embeddings = local_model.encode(texts, show_progress_bar=False)
            
            if hasattr(local_embeddings, "tolist"):
                all_embeddings = local_embeddings.tolist()
            else:
                all_embeddings = [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in local_embeddings]
                
            return all_embeddings[0] if is_single else all_embeddings
        except Exception as local_err:
            print(f"Local embedding generation also failed: {local_err}")
            raise api_err


def get_weaviate_client():
    """
    Establish a connection to the Weaviate v4 instance.
    Supports local Docker setup or Weaviate Cloud (WCD) cluster configuration.
    """
    url = get_secret("WEAVIATE_URL", "http://localhost:8080").strip()
    api_key = get_secret("WEAVIATE_API_KEY")
    
    # Auto-prepend scheme if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
            
    auth = wvc.init.Auth.api_key(api_key) if api_key else None
    
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=auth
    )

def build_vectorstore(chunks):
    """
    Encode chunks using Hugging Face Inference API and upload them to Weaviate in batch.
    Supports multi-domain storage by deleting only domain-specific documents.
    """
    # Extract texts and compute embeddings using Hugging Face API
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_huggingface_embeddings(texts)
    
    target_domain = ""
    if chunks:
        target_domain = urlparse(chunks[0]["url"]).netloc
        
    with get_weaviate_client() as client:
        collection_name = "WebsiteData"
        
        # If collection exists, verify it has the 'domain' property
        if client.collections.exists(collection_name):
            collection = client.collections.get(collection_name)
            config = collection.config.get()
            has_domain = any(prop.name == "domain" for prop in config.properties)
            
            if not has_domain:
                print(f"Collection '{collection_name}' is missing 'domain' property. Recreating collection...")
                client.collections.delete(collection_name)
                collection = client.collections.create(
                    name=collection_name,
                    properties=[
                        wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="domain", data_type=wvc.config.DataType.TEXT),
                    ],
                    vectorizer_config=None
                )
            elif target_domain:
                collection.data.delete_many(
                    where=Filter.by_property("domain").equal(target_domain)
                )
        else:
            # Create collection without server-side vectorizer (we supply vectors)
            collection = client.collections.create(
                name=collection_name,
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="domain", data_type=wvc.config.DataType.TEXT),
                ],
                vectorizer_config=None
            )
            
        # Get collection instance (handles newly created or existing)
        collection = client.collections.get(collection_name)
        
        # Batch upload to Weaviate
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                # Handle standard list or numpy array/tensor formats safely
                vector = embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else embeddings[i]
                batch.add_object(
                    properties={
                        "text": chunk["text"],
                        "url": chunk["url"],
                        "domain": target_domain
                    },
                    vector=vector
                )
        print(f"Ingested {len(chunks)} chunks for domain '{target_domain}' into Weaviate.")