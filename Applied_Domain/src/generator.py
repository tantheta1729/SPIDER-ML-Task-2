import os
from google import genai
from google.genai import types
from src.config import GEN_MODEL_NAME

class EvidenceGenerator:
    def __init__(self):
        api_key = "AQ.Ab8RN6IobQYGdpDaESDOECVdanrvWIINh4bkmq5OEO08OVDmEA"
        self.client = genai.Client(api_key=api_key)

    def rewrite_query(self, query: str, chat_history: list[dict]) -> str:
        """Rewrites queries based on context for multi-turn search."""
        if not chat_history:
            return query

        recent_history = chat_history[-4:]
        history_str = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('parts', '')}" 
            for msg in recent_history
        ])

        prompt = f"""Rewrite this follow-up question into a standalone medical search query.
History:
{history_str}
Follow-up: {query}
Standalone Query:"""

        try:

            response = self.client.models.generate_content(
                model=GEN_MODEL_NAME,
                contents=prompt
            )
            return response.text.strip() if response.text else query
        except Exception as e:
            print(f"⚠️ Rewrite failed: {e}")
            return query

    def build_system_instruction(self, evidence_docs: list[dict]) -> str:
      """Formats evidence chunks into a strict system instruction."""
      context_blocks = []
      for idx, doc in enumerate(evidence_docs, 1):
        context_blocks.append(
            f"EVIDENCE CHUNK [{idx}]\n"
            f"Source: {doc['source']} (Topic: {doc['topic']})\n"
            f"Content: {doc['content']}\n"
        )

      context_str = "\n-----\n".join(context_blocks)

      return f"""You are a trustworthy Clinical Healthcare Information Assistant.

INSTRUCTIONS:
1. Only if the user greets you (e.g., "hello", "hi", "good morning"), respond politely and invite them to ask a health-related question. Dont simply Greet for every query.
2. For medical questions, answer strictly using ONLY the evidence chunks provided below. Use inline numerical citations like [1], [2].
3. If the evidence chunks lack information to answer a medical question, state: "Insufficient evidence in verified medical guidelines to answer this question."

PROVIDED EVIDENCE CHUNKS:
{context_str}
"""

    def generate_stream(self, query: str, evidence_docs: list[dict], chat_history: list[dict] = None):
        """Streams response tokens bypassing the chats module for better stability."""
        if chat_history is None:
            chat_history = []
            
        system_instruction = self.build_system_instruction(evidence_docs)
        
        # 1. Formatting the history array for the direct models API
        contents = []
        for msg in chat_history:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("parts", ""))]
                )
            )
            
        # 2. Appending the current user query to the end of the contents
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)]
            )
        )

        try:
            # 3. Calling the stream directly
            response_stream = self.client.models.generate_content_stream(
                model=GEN_MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            # Instead of crashing the frontend, stream the actual error to the chat bubble!
            yield f"\n\n[Google API Error: {str(e)}]"