from langchain.vectorstores import Chroma
from embeddings.embedder import get_embedding_model

def get_vectorstore(documents, persist_directory="chroma_db"):
    embedding_model = get_embedding_model()
    vectordb = Chroma.from_documents(
        documents, 
        embedding_model, 
        persist_directory=persist_directory
        )
    vectordb.persist()
    return vectordb
