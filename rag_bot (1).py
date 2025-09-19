# pip install sentence-transformers groq faiss-cpu

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import json
from google.colab import userdata
import requests

# Initialize Groq client
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
llm_model = "llama-3.3-70b-versatile"

API_URL = "https://api.groq.com/openai/v1/chat/completions"

"""**Step 3: Load the Embedding Model**"""

# Load the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_dim = embedding_model.get_sentence_embedding_dimension()  # Typically 384 for this model

embedding_dim

"""**Step 4: Prepare Your Documents**"""

# Sample documents (replace with your own)
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "The Eiffel Tower is in Paris.",
    "Groq provides fast AI inference."
]

"""**Step 5: Generate Embeddings for Documents**"""

# Generate embeddings
doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')  # FAISS requires float32

"""**Step 6: Set Up FAISS Vector Database**"""

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance metric
index.add(doc_embeddings)  # Add document embeddings to the index

"""**Step 7: Define the Retrieval Function**"""

def retrieve_documents(query, top_k=3):
    # Embed the query
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)

    # print(distances, indices)

    # Retrieve the matching documents
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def generate_response(query, retrieved_docs, temperature=0.4):

    # Create context from retrieved documents
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    # Refined prompt template
    prompt = f"""
            You are a concise and accurate assistant. Use the provided context to answer the query directly and clearly. If the context doesn't contain relevant information, Then simply say **Can't provide a valid ans**.

            Context:
            {context}

            Query: {query}

            Answer:
            """

    # Prepare headers and data for Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    try:
        # Send request to Groq API
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))

        # Check response status
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content'].strip()
        else:
            return f"Error: API returned status code {response.status_code}: {response.text}"

    except requests.RequestException as e:
        return f"Error in API request: {str(e)}"

# Example usage
# query = "What is the capital of France?"

query = "What is Django?"


retrieved = retrieve_documents(query, top_k=2)
print("Retrieved Documents:", retrieved)

response = generate_response(query, retrieved)
print("Generated Response:", response)
