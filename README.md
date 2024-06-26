# Customer Support Chatbot

## Project Overview

This project involves the development of a Customer Support Chatbot utilizing advanced AI techniques and resources. The chatbot aims to provide efficient and accurate customer support by leveraging the capabilities of fine-tuned language models and retrieval-augmented generation (RAG).

## Features

- **Fine-Tuned Language Model**: Utilizes the LLAMA 3 8B model fine-tuned on customer support data.
- **Retrieval-Augmented Generation (RAG)**: Integrates RAG technique to enhance the chatbot's performance.
- **PyTorch and Colab**: Developed using PyTorch and Google Colab, leveraging its free T4 GPU.
- **Hugging Face Dataset**: Trained on the Bitext customer support dataset available on Hugging Face.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Hugging Face
- Google Colab account (for using T4 GPU)

### Dataset
The training data for this project is sourced from the Hugging Face dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset.

### Model Training
The LLAMA 3 8B model is fine-tuned using the unsloth repository on GitHub. Detailed instructions for training can be found in the notebook.

### Integrating RAG

I plan to integrate the RAG technique using LangChain. This involves retrieving relevant documents or information to augment the responses generated by the fine-tuned model.

### Future Work

Complete RAG Integration: Fully integrate the RAG technique with the fine-tuned model using LangChain.
Expand Dataset: Incorporate additional customer support datasets to improve model performance.
Deploy Application: Deploy the chatbot as a web or mobile application for real-world usage.

### Contributing

Contributions are welcome! Please open an issue or submit a pull request with your suggestions and improvements.
