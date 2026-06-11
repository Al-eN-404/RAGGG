import json

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

# pages=crawl_website("https://en.wikipedia.org/wiki/Transport", maxpages=3)

# chunks = chunk_page(pages)

# print(f"Total Pages: {len(pages)}")
# print(f"Total Chunks: {len(chunks)}")
# print("\nFirst Chunk:\n")
# print(chunks[0]["text"][:500])

# ### Save Chunks to JSON File
# with open(
#     "chunks.json",
#     "w",
#     encoding="utf-8"
# ) as f:

#     json.dump(
#         chunks,
#         f,
#         ensure_ascii=False,
#         indent=2
#     )
#################################################################
# print(
#     f"Total Chunks: {len(chunks)}"
# )

# print("\nFirst Chunk:\n")

# print(
#     chunks[0]["text"][:500]
# )