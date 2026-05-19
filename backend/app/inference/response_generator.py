_TEMPLATES: dict[str, str] = {
    "anger": "I heard frustration or anger. The words suggested {text_label}, while the tone suggested {audio_label}.",
    "disgust": "That sounded like strong disapproval. The words suggested {text_label}, while the tone suggested {audio_label}.",
    "fear": "That sounded nervous or uncertain. The words suggested {text_label}, while the tone suggested {audio_label}.",
    "joy": "That sounded positive and upbeat. The words suggested {text_label}, while the tone suggested {audio_label}.",
    "neutral": "That sounded mostly neutral. The words suggested {text_label}, while the tone suggested {audio_label}.",
    "sadness": "That sounded disappointed or sad. The words suggested {text_label}, while the tone suggested {audio_label}.",
    "surprise": "That sounded surprised. The words suggested {text_label}, while the tone suggested {audio_label}.",
}


def generate_response(fusion_label: str, text_label: str, audio_label: str) -> str:
    if fusion_label == text_label == audio_label:
        return f"Both your words and tone suggest {fusion_label}."

    template = _TEMPLATES.get(
        fusion_label,
        "The overall analysis suggests {fusion_label}. The words suggested {text_label}, while the tone suggested {audio_label}.",
    )
    return template.format(
        fusion_label=fusion_label,
        text_label=text_label,
        audio_label=audio_label,
    )
