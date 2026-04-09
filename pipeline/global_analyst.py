"""
Global Analyst Module (Gemini 2.5 Pro)
──────────────────────────────────────
Generates the Executive Summary for the global data science salary market
using Google's Gemini 2.5 Pro API.

This module handles ONLY the macro-level market synthesis. Micro-narratives
(per-record) remain on the local Ollama/Llama 3.2 instance in llm_analyst.py.

Architecture:
    Micro-narratives  → Ollama (local, fast, per-record)
    Executive summary → Gemini 2.5 Pro (cloud, advanced reasoning, one-shot)
"""

import logging
import os

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── Configure Gemini (must run after load_dotenv() in the caller) ────────────
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def generate_summary(stats: dict) -> str:
    """
    Uses Gemini 2.5 Pro to write a 3-paragraph Executive Summary of the
    global data science salary market.

    Parameters
    ----------
    stats : dict
        Pre-calculated market statistics with these keys:
        - na_premium_pct      : float — how much higher NA salaries are vs ROW
        - seniority_jump_pct  : float — % increase from Mid to Senior level
        - startup_penalty_pct : float — % penalty for seniors in small startups

    Returns
    -------
    str
        A 3-paragraph executive summary, or a fallback message on failure.
    """
    system_instruction = (
        "You are a Senior Labor Economist. Write a 3-paragraph executive "
        "summary of the global data science market.\n"
        "Format:\n"
        "Paragraph 1: Global geographic wealth gap.\n"
        "Paragraph 2: Career progression and company size dynamics.\n"
        "Paragraph 3: Outlook for the industry.\n"
        "Do not use markdown headers for the paragraphs. Just output clean text."
    )

    prompt = (
        "Use these exact statistics in your analysis:\n"
        f"- North American salaries are {stats['na_premium_pct']}% higher "
        "than the rest of the world.\n"
        f"- Moving from Mid-level to Senior results in a "
        f"{stats['seniority_jump_pct']}% salary increase on average.\n"
        f"- Seniors in small startups face a {stats['startup_penalty_pct']}% "
        "'startup penalty' compared to mid/large firms."
    )

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction=system_instruction,
        )
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        logger.error("Gemini API failed: %s", e)
        return "Market analysis temporarily unavailable."
