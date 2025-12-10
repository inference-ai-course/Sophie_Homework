# router.py
import json
from tool_registry import TOOL_REGISTRY

def route_llm_output(llm_output: str) -> str:
    """
    Routes LLM output to the correct tool if it's a function call.
    """
    try:
        output = json.loads(llm_output)
        func_name = output.get("function")
        args = output.get("arguments", {})
    except (json.JSONDecodeError, TypeError):
        # Not a function call → normal text
        return llm_output

    tool = TOOL_REGISTRY.get(func_name)

    if not tool:
        return f"Error: Unknown function '{func_name}'"

    try:
        return tool(**args)
    except Exception as e:
        return f"Tool Execution Error: {e}"
