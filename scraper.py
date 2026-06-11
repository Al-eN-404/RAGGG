import os
import requests

def crawl_website(start_url, maxpages=10):
    """
    Scrapes the target website by delegating the crawling job to our
    dedicated backend API running on Hugging Face Spaces.
    """
    # Look up the Hugging Face Scraper API backend URL
    try:
        import streamlit as st
        backend_url = st.secrets.get("HF_BACKEND_URL", "").strip()
    except Exception:
        backend_url = ""

    if not backend_url:
        backend_url = os.getenv("HF_BACKEND_URL", "https://alenja002-rag-backend.hf.space").strip()

    # Clean trailing slash if present
    if backend_url.endswith("/"):
        backend_url = backend_url[:-1]

    payload = {
        "url": start_url,
        "max_pages": maxpages
    }
    
    response = requests.post(f"{backend_url}/scrape", json=payload)
    
    if response.status_code != 200:
        raise Exception(
            f"Failed to scrape website via Hugging Face API (Status {response.status_code}): {response.text}\n"
            f"Please verify your HF_BACKEND_URL config in Streamlit secrets."
        )
        
    return response.json()