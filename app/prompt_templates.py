from typing import Dict

PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "detailed": {
        "base": "Please provide a detailed answer to the following question about this image: {question}",
        "explain": "Please provide a detailed answer to the following question about this image: {question}. Also explain how you arrived at this answer."
    },
    "casual": {
        "base": "Hey! Can you tell me about this image? Specifically: {question}",
        "explain": "Hey! Can you tell me about this image? Specifically: {question}. And could you explain how you figured that out?"
    },
    "concise": {
        "base": "Briefly answer this question about the image: {question}",
        "explain": "Briefly answer this question about the image: {question}. Also briefly explain your reasoning."
    }
}

def get_prompt_template(style: str = "detailed", explain: bool = False) -> str:
    """
    Get the appropriate prompt template based on style and explanation mode.
    
    Args:
        style: The style of the answer (detailed, casual, concise)
        explain: Whether to include explanation
    
    Returns:
        str: The prompt template
    """
    if style not in PROMPT_TEMPLATES:
        style = "detailed"  # Default to detailed if style not found
    
    template_key = "explain" if explain else "base"
    return PROMPT_TEMPLATES[style][template_key] 