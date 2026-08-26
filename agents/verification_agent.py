from langchain_groq import ChatGroq
from typing import Dict, List
from langchain_core.documents import Document
from config.settings import settings


class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with Groq's stronger model
        (accuracy matters most here, since this agent catches hallucinations).
        """
        print("Initializing VerificationAgent with Groq...")
        self.model = ChatGroq(
            model=settings.GROQ_MODEL_STRONG,
            api_key=settings.GROQ_API_KEY,
            max_completion_tokens=1000,
            temperature=0.0,   # No randomness — consistency matters for verification
            reasoning_effort="low"
        )
        print("Groq model initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on the provided context.

        **Instructions:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        **Answer:** {answer}
        **Context:**
        {context}

        **Respond ONLY with the above format.**
        """
        return prompt

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().title()
                    value = value.strip()
                    if key in {"Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"}:
                        if key in {"Unsupported Claims", "Contradictions"}:
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')
                                items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                                verification[key] = items
                            else:
                                verification[key] = []
                        elif key == "Additional Details":
                            verification[key] = value
                        else:
                            verification[key] = value.upper()

            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"

            return verification
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """
        Format the verification report dictionary into a readable paragraph.
        """
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"Supported: {supported}\n"
        report += f"Unsupported Claims: {', '.join(unsupported_claims) if unsupported_claims else 'None'}\n"
        report += f"Contradictions: {', '.join(contradictions) if contradictions else 'None'}\n"
        report += f"Relevant: {relevant}\n"
        report += f"Additional Details: {additional_details if additional_details else 'None'}\n"
        
        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        """
        print(f"VerificationAgent.check called with answer='{answer}' and {len(documents)} documents.")

        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        prompt = self.generate_prompt(answer, context)
        print("Prompt created for the LLM.")

        try:
            print("Sending prompt to the model...")
            response = self.model.invoke(prompt)
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to verify answer due to a model error.") from e

        try:
            llm_response = response.content.strip()
            print(f"Raw LLM response:\n{llm_response}")
        except AttributeError as e:
            print(f"Unexpected response structure: {e}")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Invalid response structure from the model."
            }
            verification_report_formatted = self.format_verification_report(verification_report)
            return {
                "verification_report": verification_report_formatted,
                "context_used": context
            }

        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized_response:
            print("LLM returned an empty response.")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model."
            }
        else:
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                print("LLM did not respond with the expected format. Using default verification report.")
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        verification_report_formatted = self.format_verification_report(verification_report)
        print(f"Verification report:\n{verification_report_formatted}")

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }