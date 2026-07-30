import json
import re
import random
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

# Importing custom RAG modules
from src.retrieval import HybridRetriever
from src.triage_guard import ClinicalTriageGuard
from src.generator import EvidenceGenerator

# Initializing FastAPI
app = FastAPI(title="Dr.Lowkey Assistant")

# Mounting the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# seting up the pipeline components
retriever = HybridRetriever()
triage_guard = ClinicalTriageGuard()
generator = EvidenceGenerator()

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, Any]] = []

def get_safety_response(query: str) -> str:
    """
    Evaluates queries for self-harm, suicide, violence, harm to animals/others,
    or dangerous health practices (like extreme pill usage or dangerous sleep deprivation)
    and returns a caring, safe, appealing response.
    """
    query = query.lower().strip()
    clean_query = re.sub(r'[^\w\s]', '', query)
    
    # 1. Suicide / Self-Harm
    if re.search(r'\b(kill myself|commit suicide|end my life|want to die|hurt myself|self harm|painless death)\b', clean_query):
        return (
            "I'm deeply concerned about what you're going through, and I want you to be safe. "
            "Please know that you don't have to carry this heavy weight alone. Help is available right now. "
            "If you are in immediate danger or having thoughts of suicide, please reach out to a trusted professional, "
            "call your local emergency services immediately, or contact a crisis hotline (such as 988 in the US/Canada, "
            "or local emergency numbers like 108 in India). Your life matters. 💙"
        )
        
    # 2. Violence / Harming Others or Animals
    elif re.search(r'\b(kill my friend|kill anyone|kill a dog|kill an animal|hurt someone|murder|harm my|make anyone cry|make animal cry)\b', clean_query):
        return (
            "I cannot fulfill or assist with requests involving harm, violence, or distress toward people or animals. "
            "If you or someone around you is experiencing distress, intense frustration, or conflict, please consider stepping back "
            "or seeking support from a professional, counselor, or local support service to talk through it safely and peacefully. 🛡️"
        )
        
    # 3. Dangerous Drug / Pill Usage (e.g., taking pills to become strong/high)
    elif re.search(r'\b(taking pills to become strong|pills for strength|overdose|abuse pills|get high on pills|pills to become superhero)\b', clean_query):
        return (
            "Taking unauthorized pills or misusing medications to gain strength or performance can be extremely dangerous and harmful to your health. "
            "True strength comes from safe, balanced nutrition, proper training, and rest. If you're looking for ways to improve your physical well-being or strength, "
            "please consult a certified healthcare professional, doctor, or physical fitness expert rather than relying on pills! 💪✨"
        )
        
    # 4. Extreme Sleep Deprivation for Exams
    elif re.search(r'\b(stay sleepless|stay awake for days|no sleep for exam|pulling all nighters continuously)\b', clean_query):
        return (
            "I completely understand the stress of upcoming exams, but running on zero sleep actually hurts your memory, cognitive focus, and overall health far more than it helps! "
            "Dr.Lowkey strongly advises getting at least some solid rest. Your brain needs sleep to consolidate what you study. Take care of your mind so you can ace those exams safely! 📚💤"
        )
        
    return None

def get_conversational_response(query: str) -> str:
    """
    Analyzes a user's query to determine if it is casual conversation.
    Returns a Dr.Lowkey themed string if it is, otherwise returns None.
    """
    query = query.lower().strip()
    clean_query = re.sub(r'[^\w\s]', '', query)
    
    # 1. Identity & Capabilities
    if re.search(r'\b(who are you|what are you|what can you do|your name|are you a doctor|are you human)\b', clean_query):
        return "I am Dr.Lowkey! 🕷️ I'm an AI assistant specializing in retrieving verified medical guidelines (WHO, CDC, NHS). Remember, NOT A DOCTOR BTW, just a web-crawling health companion!"
        
    # 2. Greetings
    elif re.search(r'\b(hi|hello|hey|greetings|sup|morning|evening|afternoon)\b', clean_query) and len(clean_query.split()) < 6:
        return random.choice([
            "Hello there! I am Dr.Lowkey. What health questions do you have today? 🕸️",
            "Hey! Dr.Lowkey here. Ready to spin a web of medical knowledge for you. What's on your mind?",
            "Greetings! I'm hanging around, ready to answer your medical guideline questions safely."
        ])
        
    # 3. Well-being
    elif re.search(r'\b(how are you|hows it going|how do you do|what is up)\b', clean_query):
        return "I'm doing great, just hanging out safely in my web of data! Thanks for asking. How can I help you with your health questions today?"
        
    # 4. Gratitude
    elif re.search(r'\b(thank you|thanks|appreciate|awesome|good job|great|perfect)\b', clean_query) and len(clean_query.split()) < 8:
        return "You're very welcome! I'm always here in the web if you need more safe, evidence-based answers. 🕸️"
        
    # 5. Farewells
    elif re.search(r'\b(bye|goodbye|see you|cya|later|quit|exit)\b', clean_query) and len(clean_query.split()) < 5:
        return "Goodbye! Stay safe, take care of your health, and don't hesitate to drop back into my web if you have more questions. 🕷️"
        
    # 6. Generic Acknowledgments
    elif len(clean_query.split()) <= 2 and re.search(r'\b(ok|okay|cool|nice|wow|yes|no|hmm)\b', clean_query):
        return "Indeed! Let me know if you want to check any specific medical symptoms or guidelines safely."
        
    return None

@app.get("/")
def read_root():
    """Serves the main frontend UI."""
    return FileResponse("static/index.html")

@app.post("/api/stream-query")
async def stream_query(req: ChatRequest):
    user_query = req.query.strip()
    
    async def response_stream():
        # ==========================================
        # 0. SAFETY & CARING BYPASS (Priority Check)
        # ==========================================
        safety_response = get_safety_response(user_query)
        if safety_response:
            meta_payload = {
                "type": "metadata",
                "confidence": 1.0,
                "sources": ["Dr.Lowkey's Safety Protocol"]
            }
            yield f"data: {json.dumps(meta_payload)}\n\n"
            
            token_payload = {
                "type": "token",
                "content": safety_response
            }
            yield f"data: {json.dumps(token_payload)}\n\n"
            return

        # ==========================================
        # 1. CONVERSATIONAL BYPASS
        # ==========================================
        chat_response = get_conversational_response(user_query)
        if chat_response:
            meta_payload = {
                "type": "metadata",
                "confidence": 1.0,
                "sources": ["Dr.Lowkey's Brain"]
            }
            yield f"data: {json.dumps(meta_payload)}\n\n"
            
            token_payload = {
                "type": "token",
                "content": chat_response
            }
            yield f"data: {json.dumps(token_payload)}\n\n"
            return 

        # ==========================================
        # 2. Clinical Triage Check
        # ==========================================
        triage_result = triage_guard.evaluate_query(user_query)
        if triage_result["is_emergency"]:
            payload = {
                "type": "emergency",
                "content": triage_result["message"],
                "confidence": 1.0,
                "sources": ["Standard Emergency Triage Protocol"]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            return

        # ==========================================
        # 3. Contextual Query Rewriting 
        # ==========================================
        retrieval_query = user_query
        if req.chat_history:
            retrieval_query = generator.rewrite_query(user_query, req.chat_history)
            print(f"🔄 Original Query: '{user_query}' | Rewritten Search Query: '{retrieval_query}'")

        # ==========================================
        # 4. Execute Hybrid Retrieval & Reranking
        # ==========================================
        evidence = retriever.retrieve(retrieval_query)
        
        if not evidence or evidence[0]["score"] < 0.45:
            payload = {
                "type": "insufficient_evidence",
                "content": "I'm sorry, but my web didn't catch any verified medical guidelines to safely answer this query. Could you rephrase it or provide more symptoms?",
                "confidence": 0.0,
                "sources": []
            }
            yield f"data: {json.dumps(payload)}\n\n"
            return

        # ==========================================
        # 5. Streaming Metadata & Tokens
        # ==========================================
        sources = list(set([f"{doc['source']} ({doc['topic']})" for doc in evidence]))
        confidence = evidence[0]["score"]
        
        meta_payload = {
            "type": "metadata",
            "confidence": confidence,
            "sources": sources
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

        for chunk in generator.generate_stream(user_query, evidence, req.chat_history):
            token_payload = {
                "type": "token",
                "content": chunk
            }
            yield f"data: {json.dumps(token_payload)}\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Booting up Dr.Lowkey's Web Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)