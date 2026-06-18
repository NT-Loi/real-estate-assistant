import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from rag.chain import RAGChain

def main():
    chain = RAGChain()
    
    query = "Dùng web_search tra bài đánh giá dự án Akari City Bình Tân, sau đó chọn 1 URL để dùng read_url đọc chi tiết rồi tóm tắt lại."
    print(f"\n--- QUERY: {query} ---\n")
    
    response = chain.query(query)
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(response.answer)
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
