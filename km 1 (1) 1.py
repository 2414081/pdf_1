
import os
import logging
import uuid  # For generating unique IDs
import pandas as pd
import requests
from src.GenerativeAI.DBOperation.DatasetReadWrite import datasetReadWrite
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.Controllers.ServiceController import app
from flask import json
from litellm import completion
os.environ["AZURE_OPENAI_API_KEY"] = "6ffhCm6wvgxD2LRD7wNZI5sHgqHn4lqYOY7xn8Ycdg7vHvZ8qyujJQQJ99BCACYeBjFXJ3w3AAABACOGwuFA"
os.environ["AZURE_API_BASE"] = "https://tcoeaiteamgpt4o.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"
from src.GenerativeAI.CoreLogicLayer.Evaluation.EvaluationAsync import EvaluationAsync
# Set up logging for detailed output
# logging.basicConfig(level=logging.DEBUG)
# from src.GenerativeAI.LLMLayer.LLMInitializer import LLMInterface,get_llm_response
# # LLMInterface("67909e402122c7d113d6d725")
# # Function to initialize the model based on user choice
from langfuse import Langfuse
evaluation_processor=EvaluationAsync()
mongo_obj = datasetReadWrite()
langfuse = Langfuse(
    secret_key="sk-lf-22dda636-e3bc-40db-b766-d19940bc766c",
    public_key="pk-lf-f7995fdb-81cf-48f0-8260-c9d4e964fa86",
    host="https://cloud.langfuse.com"
)
baseurl="https://cloud.langfuse.com"
api_key="sk-lf-22dda636-e3bc-40db-b766-d19940bc766c"
response = requests.get(f"{baseurl}/projects", headers={"Authorization": f"Bearer {api_key}"})

traces=langfuse.fetch_trace("f5d814c2-077b-4b52-b2bb-c2c4a4aaee9b")
print(traces)
# question = traces.input.Questions  # Extracting question
# answer = traces.output.Answer  # Extracting answer
question = traces.data.input.get("Questions")
answer=traces.data.output.get("Answer")

# Print results
print("Question:", question)
print("\nAnswer:\n", answer)



# User selects model version
model_choice = input("Select the model version (gpt-3.5/gpt-4): ").strip().lower()


# Load Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the LLM prompt
prompt_template = """
You are an intelligent assistant. Answer the user's question based on the provided context.

Context:
{context}

User Question:
{question}
"""


# Generate a unique user_id when the session starts
user_id = str(uuid.uuid4())
print(f"New session started for user_id: {user_id}")


# Document upload and chunking
def upload_and_chunk_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


# Generate embeddings for the document chunks
def generate_chunk_embeddings(chunks):
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(chunk_texts, convert_to_tensor=True)
    return chunk_texts, embeddings


# Load and process the document
file_path = r"C:\Users\SPRSVTVFST\Downloads\MLOps Level 1.pdf"
chunks = upload_and_chunk_document(file_path)
chunk_texts, chunk_embeddings = generate_chunk_embeddings(chunks)

while True:
    # Generate a new session_id for each interaction
    session_id = str(uuid.uuid4())
    print(f"New session_id for this interaction: {session_id}")

    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print(f"Session ended for user_id: {user_id}")
        print("Exiting the chat. Goodbye!")
        break

    user_embedding = embedding_model.encode([user_input], convert_to_tensor=True)[0]

    # Calculate similarities between user input and chunk embeddings
    similarities = cosine_similarity([user_embedding], chunk_embeddings)
    top_k_indices = similarities[0].argsort()[-2:][::-1]  # Get indices of top 2 similar chunks

    relevant_chunks = [chunk_texts[i] for i in top_k_indices]
    similarity_scores = similarities[0][top_k_indices]

    print("Relevant Chunks Used for Context:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"Chunk {i + 1}: {chunk}")

    if all(similarity < 0.3 for similarity in similarity_scores):
        print("Bot: I do not have enough relevant context to answer this question.")
        continue

    context_input = ' '.join(relevant_chunks)

    # Format the LLM input
    llm_input = prompt_template.format(context=context_input, question=user_input)

    try:
        # llm_response = get_llm_response("Gpt-4o",llm_input)
        llm_response= completion(
            model="azure/gpt-4o",
            messages=[{"content": llm_input, "role": "user"}]
        )
        answer = llm_response["choices"][0]["message"]["content"]
        print(f"Bot: {answer}")
        trace_ids=["f5d814c2-077b-4b52-b2bb-c2c4a4aaee9b"]
        input_param = {
           "traces":trace_ids,

            "status": 0,
            "executionType": "genAIEvaluation",
            "configId": "67aae9a17a086e172e1cfec8",
            "executedBy": "testUser",
            "projectName": "Observability",
            "datasetId": "681c6f83b42e2b7f1c51953a",
            "model": "Gpt-4o",
            "totalRecords":2,
            "projectType": "Observability",
            "jobID": str(uuid.uuid4())  # Generate a unique job ID
        }

        try :# Call the endpoint
            def call_internal_endpoint():
                with app.test_client() as client:
                    payload = input_param
                    response = client.post("/eval", json=payload)

                    return response
        except Exception as e:
            logging.error(e)
            print(e)
        data =call_internal_endpoint()
        # evaluation_result = evaluation_processor.evaluation_calculation(input_param)
        # print(f"Evaluation Result: {evaluation_result}")

        record = {
            "question": user_input,
            "context": context_input,
            "answer": answer,
            "session_id": str(uuid.uuid4())
        }
        mongo_obj.write_single_data('Multiple',record)
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        print("Bot: Sorry, there was an issue generating the response.")
    trace = langfuse.trace(
        name="Testing ",
        input={"Questions": user_input},
        output={"Answer":answer},
        metadata={"run_name": record}
    )