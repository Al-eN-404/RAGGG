
from scraper import crawl_website
from langchain_text_splitters import (RecursiveCharacterTextSplitter)

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunks=[]
    
    for page in pages:
        page_chunk=splitter.split_text(page["text"])
        
        for chunk in page_chunk:
            chunks.append({"url":page["url"], "text":chunk})
    
    return chunks


