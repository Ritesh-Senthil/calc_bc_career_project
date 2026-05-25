_TEMPLATES: dict[str, str] = {
    "anger": "That reads as frustration or anger.",
    "disgust": "That reads as strong disapproval or disgust.",
    "fear": "That reads as nervous or uncertain.",
    "joy": "That reads as positive and upbeat.",
    "neutral": "That reads as mostly neutral.",
    "sadness": "That reads as disappointed or sad.",
    "surprise": "That reads as surprised.",
}


def generate_response(label: str, *_unused) -> str:
    return _TEMPLATES.get(label, f"The analysis suggests {label}.")
