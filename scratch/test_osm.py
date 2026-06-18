import asyncio
import os
import sys

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chain import RAGChain

def test_osm():
    chain = RAGChain()
    query = "Tìm các căn hộ bán kính 2km quanh Landmark 81"
    
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        # Use query_stream to see the reasoning steps
        for event in chain.query_stream(query):
            if event["type"] == "status":
                print(f"[STATUS] {event['text']}")
            elif event["type"] == "reasoning":
                print(f"[REASONING] {event['text']}")
            elif event["type"] == "chunk":
                print(event["text"], end="", flush=True)
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    test_osm()
