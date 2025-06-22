from loaders.load_data import load_documents
from vectorstore.chroma_store import get_vectorstore
from rag_pipeline.qa_chain import get_qa_chain

def main():
    print("🔄 Loading documents...")
    documents = load_documents()
    
    print("Creating vector store...")
    vectordb = get_vectorstore(documents)

    print("Starting QA system...")
    qa_chain = get_qa_chain(vectordb)

    while True:
        query = input("\nAsk your question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        result = qa_chain({"query": query})
        print(f"Answer: {result['result']}")

if __name__ == "__main__":
    main()
