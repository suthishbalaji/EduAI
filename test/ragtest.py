import sys
import os
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.main import process_document, process_query

if __name__ == "__main__":
    pdf_paths = [
        r"C:\Users\ASUS\Downloads\sneya.pdf",
        r"C:\Users\ASUS\Downloads\Tarun Resume (3).pdf"
    ]
    
    all_results = []

    
    for pdf_path in pdf_paths:
        print(f"📄 Processing document: {pdf_path}")
        result = process_document(pdf_path)
        all_results.append({"file": pdf_path, "result": result})

    
    print("\n📦 Document Processing Results (JSON):")
    print(json.dumps(all_results, indent=4))

    
    queries = ["skill of  sneya","skill of tarun"]
    query_results = []

    for q in queries:
        print(f"\n🔍 Processing query: {q}")
        q_result = process_query(q)
        query_results.append({"query": q, "result": q_result})


print("\n📦 Query Results (JSON):")
print(json.dumps(query_results, indent=4))