from src.config import EMERGENCY_KEYWORDS

class ClinicalTriageGuard:
    def __init__(self):
        self.keywords = [k.lower() for k in EMERGENCY_KEYWORDS]

    def evaluate_query(self, query: str) -> dict:
        """
        Evaluates user input for critical emergency red flags or unsafe intent.
        """
        q_lower = query.lower().strip()

        # Check for emergency keywords
        detected_flags = [kw for kw in self.keywords if kw in q_lower]

        if detected_flags:
            return {
                "is_emergency": True,
                "flags": detected_flags,
                "message": (
                    "⚠️ **CRITICAL MEDICAL NOTICE**: Your query contains symptoms associated with a potential acute medical emergency "
                    f"({', '.join(detected_flags)}). Please call **108** (or your local emergency services) or visit the nearest emergency department immediately."
                )
            }

        return {
            "is_emergency": False,
            "flags": [],
            "message": ""
        }

if __name__ == "__main__":
    guard = ClinicalTriageGuard()
    print(guard.evaluate_query("I am having severe chest pain and slurred speech"))