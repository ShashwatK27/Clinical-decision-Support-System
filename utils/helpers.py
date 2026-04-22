"""
Shared utility helpers for CDSS.
"""


def sanitize_log_text(text: str) -> str:
    """
    Return a safe summary of a text string for logging, avoiding exposure
    of Protected Health Information (PHI) such as raw prescription text.

    Instead of logging the raw content, we log only its length and
    word count.

    Args:
        text: The original text to sanitize.
    Returns:
        A safe, non-identifying string description of the text.
    """
    if not text:
        return "[empty]"
    word_count = len(text.split())
    char_count = len(text)
    return f"[{char_count} chars, {word_count} words]"
