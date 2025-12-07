def get_personalized_content(content: str, background: str) -> str:
    """
    This function will take the content and the user's background
    and return a personalized version of the content.
    """
    if "beginner" in background.lower():
        # Simple logic: if the user is a beginner, add a note.
        return f"[BEGINNER FRIENDLY]\n{content}"
    return content

