import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
from src.crawler.scraper import FastAPIScraper
from src.crawler.text_processor import TextProcessor
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
from src.config.settings import settings


async def main():
    print("Starting FastAPI Documentation Knowledge Base Build...")
    
    # Step 1: Crawl FastAPI docs
    print("Step 1: Crawling FastAPI documentation...")
    scraper = FastAPIScraper()
    pages = await scraper.crawl(max_pages=50)  # Increased for more coverage
    print(f"Successfully crawled {len(pages)} pages.")
    
    # Step 2: Process and chunk text
    print("Step 2: Processing and chunking text...")
    processor = TextProcessor()
    all_chunks = []
    all_metadata = []
    
    for page in pages:
        try:
            main_text = processor.extract_main_content(page.html)
            clean_text = processor.clean_text(main_text)
            
            if clean_text.strip():  # Only process non-empty content
                chunks = processor.chunk_text(clean_text)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk.text)
                    all_metadata.append({
                        'url': page.url,
                        'title': f"FastAPI Docs - {page.url.split('/')[-1] or 'Home'}",
                        'chunk_index': i,
                        'source': 'fastapi_docs'
                    })
        except Exception as e:
            print(f"Error processing {page.url}: {e}")
            continue
    
    print(f"Created {len(all_chunks)} text chunks from {len(pages)} pages.")
    
    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    embeddings = embedding_service.embed_texts(all_chunks)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Step 4: Build and save vector store
    print("Step 4: Building vector database...")
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_vectors(embeddings, all_chunks, all_metadata)
    
    # Save the vector store
    os.makedirs(os.path.dirname(settings.VECTOR_DB_PATH), exist_ok=True)
    vector_store.save_index(settings.VECTOR_DB_PATH)
    print(f"Vector database saved to {settings.VECTOR_DB_PATH}")
    
    # Step 5: Display statistics
    stats = vector_store.get_stats()
    print("\n=== Knowledge Base Statistics ===")
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Vector dimension: {stats['dimension']}")
    print(f"Index type: {stats['index_type']}")
    print(f"Embedding model: {embedding_service.model_name}")
    print("Knowledge base build completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
