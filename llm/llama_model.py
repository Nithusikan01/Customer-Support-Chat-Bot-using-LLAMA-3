import os
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()

def load_llama_model():
    model_name = os.getenv("MODEL_NAME")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype="auto"
        )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512
        )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
