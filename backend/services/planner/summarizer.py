"""
Standalone summarization helpers.

- summarize_chat: legacy wrapper (still returns a disabled notice)
- summarize_text: summarize arbitrary text using a clean, ephemeral model run
"""
from __future__ import annotations

from backend.services.connectors.llama_session_manager import LlamaSessionManager


def summarize_text(
    text: str,
    session_mgr: LlamaSessionManager,
    *,
    max_tokens: int = 256,
) -> str:
    """
    Summarize arbitrary text using a dynamic token cap based on input length.
    """
    words = text.split()
    word_count = len(words)

    # Target summary length: ~60% of source words, minimum 20 words.
    target_words = max(20, int(word_count * 0.6))
    # Convert to token budget with a small safety factor.
    dynamic_max_tokens = min(max_tokens, int(target_words * 1.5))

    system_instruction = """
You are a summarizer. Follow these rules:

- Write a single unified paragraph
- Include both the main idea and essential supporting points
- The summary must be shorter than the original
- No headings, no bullet points
- No chain-of-thought, no explanations, no meta commentary
- Do not mention summary length
SUMMARY_END marks the end of your output.
""".strip()

    user_prompt = f"""
Summarize the following text:

{text}

Write the summary, then end with SUMMARY_END.
""".strip()

    summary = session_mgr.run_ephemeral(
        system_prompt=system_instruction,
        user_prompt=user_prompt,
        max_tokens=dynamic_max_tokens,
        temperature=0.2,
    )
    return summary.replace("SUMMARY_END", "").strip()


__all__ = ["summarize_text"]
