from sentence_transformers import SentenceTransformer
from vectorstore import get_weaviate_client

# Initialize the embedding model globally for fast retrieval response times
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(query, k=3):
    # Encode query locally to get the embedding vector
    query_embedding = model.encode(query)
    
    retrieved_chunks = []
    
    with get_weaviate_client() as client:
        collection_name = "WebsiteData"
        
        # Get collection object
        collection = client.collections.get(collection_name)
        
        # Query Weaviate using the embedded query vector
        response = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=k,
            return_properties=["text", "url"]
        )
        
        for obj in response.objects:
            retrieved_chunks.append({
                "url": obj.properties.get("url", ""),
                "text": obj.properties.get("text", "")
            })
            
    return retrieved_chunks

if __name__=="__main__":
    query = "What is transport?"
    try:
        results = retrieve(query)
        for i, chunk in enumerate(results):
            print(f"\nResult {i+1}")
            print(chunk["url"])
            print(chunk["text"][:300])
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        print("Please ensure your Weaviate instance is running and configured correctly.")
