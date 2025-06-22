from langchain.chains import RetrievalQA
from llm.llama_model import load_llama_model

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = load_llama_model()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa
