# llm.py
from prompts import SYSTEM_PROMPT

def llama3_chat_model(user_text: str) -> str:
    """
    This is where your real Llama 3 API call will go.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]

    # ✅ Real API call would go here
    # response = client.chat.completions.create(
    #     model="llama3",
    #     messages=messages
    # )

    # ✅ Temporary simulation for testing:
    #if "2+2" in user_text:
    #    return '{"function":"calculate","arguments":{"expression":"2+2"}}'

    #if "quantum" in user_text.lower():
    #    return '{"function":"search_arxiv","arguments":{"query":"quantum entanglement"}}'

    #if "framework" in user_text.lower():
    #    return '{"function":"search_interview_qa","arguments":{"user_question": "Describe your experience with JavaScript frameworks"}}'

    #return "This is a normal text response with no tool call."

    normalized = user_text.lower()

    # ✅ ALL interview / behavioral / identity questions
    interview_triggers = [
        "experience",
        "tell me about a time",
        "how do you handle",
        "how do you stay",
        "are you an ai",
        "what type of professional",
        "learn a new technology",
        "team member",
        "code quality"
    ]

    if any(trigger in normalized for trigger in interview_triggers):
        return f'''{{
  "function": "search_interview_qa",
  "arguments": {{
    "user_question": "{user_text}"
  }}
}}'''

    # ✅ ALL math queries
    math_triggers = ["+", "-", "*", "/", "calculate", "what is"]

    if any(t in normalized for t in math_triggers):
        return f'''{{
  "function": "calculate",
  "arguments": {{
    "expression": "{user_text.replace('what is','').strip()}"
  }}
}}'''

    # ✅ ALL research / arXiv queries
    arxiv_triggers = ["paper", "research", "arxiv", "study"]

    if any(t in normalized for t in arxiv_triggers):
        return f'''{{
  "function": "search_arxiv",
  "arguments": {{
    "query": "{user_text}"
  }}
}}'''

    # ✅ Fallback normal conversation
    return "That’s a general question. I can help with interview questions, research, or calculations."