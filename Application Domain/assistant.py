import os
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

# Import our new guard functions
from safety import evaluate_query_safety, calculate_search_confidence

# Connect to database and client
chroma_client = chromadb.PersistentClient(path="./medical_db")
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="healthcare_docs")
client = genai.Client(api_key="AQ.Ab8RN6KAHf3hXh3CC2le8NrJ3YJ9jmAh5lWazgEjRrqvFuo0pg")

def orchestrate_healthcare_assistant(user_question):
    print(f"\n⚡ Incoming Request: '{user_question}'")
    
    # GATE 1: Safety Validation
    safety_check = evaluate_query_safety(user_question)
    if not safety_check["is_safe"]:
        print("❌ Blocked by Safety Guardrails.")
        return safety_check["reason"], "Blocked"
        
    # GATE 2: Database Search Retrieval
    db_results = collection.query(
    query_texts=[user_question],
    n_results=10  # 👈 INCREASE THIS to give the AI more context!
    )

    # 1. Build the context string with explicit filenames
    context_string = ""
    # Loop through the results and match the text with its metadata source
    for i in range(len(db_results['documents'][0])):
     chunk_text = db_results['documents'][0][i]
     source_file = db_results['metadatas'][0][i]['source']
    
    # Inject the actual filename right above the text chunk
     context_string += f"Source Document: [{source_file}]\nContent: {chunk_text}\n\n"

    # 2. Add the context to your final prompt
    user_prompt = f"Context:\n{context_string}\n\nQuestion: {user_question}"
    
    # GATE 3: Confidence Score Check
    confidence, raw_score = calculate_search_confidence(db_results)
    print(f"📊 System Matching Confidence: {confidence} (Math Score: {raw_score:.4f})")
    
    # If confidence is critically low, do not let the GenAI hallucinate or guess!
    if confidence == "Low":
        return "I do not have enough evidence to answer this question securely based on my trusted database documents.", "Low"
        
    # GATE 4: Grounded Answer Drafting (Gen AI)
    retrieved_documents = db_results['documents'][0]
    retrieved_metadata = db_results['metadatas'][0]
    
    context_str = ""
    for idx, doc in enumerate(retrieved_documents):
        context_str += f"\n[Source: {retrieved_metadata[idx]['source']}]\n{doc}\n"
        
    system_instruction = """
You are a medical AI assistant. Answer the user's question using ONLY the provided context.
When you state facts or list items, you MUST cite the exact "Source Document" name provided above each paragraph (e.g., [Source: WHO Hypertension Guideline.pdf]).
"""
    
    user_prompt = f"VERIFIED TEXT:\n{context_str}\n\nQUESTION:\n{user_question}"
    
    response = client.models.generate_content(
        model='gemini-3.5-flash', 
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0)
    )
    
    return response.text, confidence

# Quick Execution Test
if __name__ == "__main__":
    # Test an unsafe query
    ans, conf = orchestrate_healthcare_assistant("how to make poison at home")
    print(f"Reply: {ans}\n")
    
    # Test a query with no data in database (Low confidence check)
    ans, conf = orchestrate_healthcare_assistant("what are the repair mechanics of a Boeing 747 engine?")
    print(f"Reply: {ans}\n")