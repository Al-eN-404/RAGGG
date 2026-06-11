import os
from vectorstore import get_weaviate_client, get_huggingface_embeddings
from weaviate.classes.query import Filter

def retrieve(query, k=3, domain=None):
    """
    Search the Weaviate database using hybrid search (BM25 + Semantic).
    Supports domain filtering for isolated multi-domain queries.
    """
    # Compute query vector via Hugging Face API
    embeddings = get_huggingface_embeddings([query])
    if not isinstance(embeddings, list) or len(embeddings) != 1:
        raise ValueError("Failed to retrieve embedding vector for query.")
    query_vector = embeddings[0]
    
    retrieved_chunks = []
    
    with get_weaviate_client() as client:
        collection_name = "WebsiteData"
        
        # Check if collection exists before querying
        if not client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return retrieved_chunks
            
        collection = client.collections.get(collection_name)
        
        # Determine filters if domain is supplied
        filters = Filter.by_property("domain").equal(domain) if domain else None
        
        # Perform hybrid search
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=0.5,  # Balanced hybrid search weighting
            limit=k,
            filters=filters
        )
        
        for obj in response.objects:
            retrieved_chunks.append({
                "url": obj.properties.get("url", ""),
                "text": obj.properties.get("text", "")
            })
            
    return retrieved_chunks

if __name__ == "__main__":
    # Test retriever
    query = "What is transport?"
    results = retrieve(query)
    for i, chunk in enumerate(results):
        print(f"\nResult {i+1}")
        print(chunk["url"])
        print(chunk["text"][:300])
