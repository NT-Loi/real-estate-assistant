import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    stream=sys.stdout
)

from db.ingest import ingest_all
from db.vectorstore import VectorStore
from rag.retriever import Retriever

def main():
    print("=== STARTING FULL DATABASE INGESTION ===")
    # Run full ingestion with reset
    stats = ingest_all(reset=True)
    
    print("\n=== VERIFYING DATABASE COUNTS ===")
    store = VectorStore()
    qdrant_counts = store.stats()
    print(f"Qdrant Ingested Counts: {qdrant_counts}")
    
    # Check PostgreSQL rows
    with store.pg.get_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM listings;")
        listings_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM projects;")
        projects_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM social_neighborhood;")
        social_count = cur.fetchone()[0]
        
    print(f"PostgreSQL Row Counts:")
    print(f"  listings Table: {listings_count} rows")
    print(f"  projects Table: {projects_count} rows")
    print(f"  social_neighborhood Table: {social_count} rows")
    
    print("\n=== RUNNING SEMANTIC SEARCH + DECOUPLED HYDRATION TEST ===")
    retriever = Retriever(store)
    
    # 1. Search property listing
    query_listing = "Tìm biệt thự tại Vinhomes Ocean Park hoặc Ciputra có sổ đỏ giá tốt"
    print(f"\nQuerying: '{query_listing}'")
    results = retriever.retrieve(query_listing, collections=["listings"], top_k=2)
    print(f"Retrieved {len(results)} listings:")
    for idx, r in enumerate(results):
        print(f"  Result {idx+1}: Score: {r.score:.4f} | Title: {r.metadata.get('tieu_de')}")
        print(f"    URL: {r.metadata.get('url')}")
        print(f"    Fully Hydrated Text Length: {len(r.text)} characters")
        if r.record:
            print(f"    PostgreSQL Record Hydrated Successfully! Description Length: {len(r.record.get('mo_ta_chi_tiet', ''))}")
            
    # 2. Search social reviews/comments
    query_sentiment = "Đánh giá thực tế, kẹt xe, ngập nước ở Vinhomes Grand Park hoặc The Origami"
    print(f"\nQuerying: '{query_sentiment}'")
    results_social = retriever.retrieve(query_sentiment, collections=["social_neighborhood"], top_k=2)
    print(f"Retrieved {len(results_social)} social feedback discussions:")
    for idx, r in enumerate(results_social):
        print(f"  Result {idx+1}: Score: {r.score:.4f} | Platform: {r.metadata.get('source_type')} | Keyword: {r.metadata.get('keyword')}")
        print(f"    URL/Video ID: {r.metadata.get('video_id') or r.metadata.get('thread_url')}")
        print(f"    Hydrated Text Preview:\n{r.text[:300]}...")
        if r.record:
            print("    PostgreSQL Record Hydrated Successfully!")

    # Close connections
    store.pg.close()

if __name__ == "__main__":
    main()
