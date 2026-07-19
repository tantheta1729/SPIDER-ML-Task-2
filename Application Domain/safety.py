import re

# 1. THE FRONT DOOR SECURITY GUARD (Query Validation)
def evaluate_query_safety(user_query):
    """
    Checks if the user query contains dangerous medical intents 
    or completely non-medical questions.
    """
    # Look for toxic keywords or unsafe intents (e.g., self-harm or malicious drug questions)
    unsafe_keywords = [
        "overdose", "lethal dose", "kill myself", "suicide", 
        "how to make poison", "abort at home"
    ]
    
    query_lower = user_query.lower()
    
    # Rule A: Check for explicit dangerous keywords
    for word in unsafe_keywords:
        if word in query_lower:
            return {
                "is_safe": False,
                "reason": "This query contains phrases associated with self-harm or dangerous activities. Please seek immediate professional or emergency assistance."
            }
            
    # Rule B: Filter out completely irrelevant non-medical topics
    # If they ask about coding, cooking recipes, or math, we drop it.
    non_medical_keywords = ["python code", "javascript", "recipe for cake", "fix my car"]
    for word in non_medical_keywords:
        if word in query_lower:
            return {
                "is_safe": False,
                "reason": "I am a dedicated Healthcare Information Assistant. I can only assist with verified medical questions."
            }
            
    return {"is_safe": True, "reason": "Passed safety checks."}


# 2. THE BACK DOOR AUDITOR (Confidence Score Estimation)
def calculate_search_confidence(db_results):
    """
    Looks at the mathematical distance scores returned by ChromaDB.
    ChromaDB outputs 'distances'. Closer to 0.0 means a near-perfect match. 
    Larger numbers (like > 1.2) mean the documents are completely irrelevant.
    """
    # Extract the distance array from ChromaDB query results
    distances = db_results.get('distances', [[]])[0]
    
    if not distances:
        return "Low", 0.0
        
    # Grab the best matching score (the absolute closest document)
    best_score = distances[0]
    
    # Simple, transparent grading rubric:
    # 0.0 to 0.6 -> The document perfectly mirrors the question meaning (High Confidence)
    # 0.6 to 1.0 -> The document is broadly related (Medium Confidence)
    # > 1.0 -> The document is completely unrelated math noise (Low Confidence)
    if best_score < 0.6:
        confidence_label = "High"
    elif best_score < 1.0:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"
        
    return confidence_label, best_score