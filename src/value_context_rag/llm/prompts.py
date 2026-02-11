"""Prompt builders for Gemma 3 12B."""

from __future__ import annotations

from typing import Iterable, List, Optional

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


VALUE_NAMES = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
]


# One-line definition for each of the 19 refined values in Table 1 of the paper “Refining the Theory of Basic Individual Values” (Journal of Personality & Social Psychology, 2012).
VALUE_DEFINITIONS = {
    "Self-direction: thought": "Freedom to cultivate one’s own ideas and abilities",
    "Self-direction: action": "Freedom to determine one’s own actions",
    "Stimulation": "Excitement, novelty, and change",
    "Hedonism": "Pleasure and sensuous gratification",
    "Achievement": "Success according to social standards",
    "Power: dominance": "Power through exercising control over people",
    "Power: resources": "Power through control of material and social resources",
    "Face": "Maintaining one’s public image and avoiding humiliation",
    "Security: personal": "Safety in one’s immediate environment",
    "Security: societal": "Safety and stability in the wider society",
    "Tradition": "Maintaining and preserving cultural, family, or religious traditions",
    "Conformity: rules": "Compliance with rules, laws, and formal obligations",
    "Conformity: interpersonal": "Avoidance of upsetting or harming other people",
    "Humility": "Recognising one’s insignificance in the larger scheme of things",
    "Benevolence: caring": "Devotion to the welfare of in-group members",
    "Benevolence: dependability": "Being a reliable and trustworthy member of the in-group",
    "Universalism: concern": "Commitment to equality, justice, and protection for all people",
    "Universalism: nature": "Preservation of the natural environment",
    "Universalism: tolerance": "Acceptance and understanding of those who are different from oneself",
}


TASK_DESCRIPTION = (
    "You are a classifier for human values in sentences. "
    "Given a TARGET SENTENCE and its context, identify which Schwartz values are present."
)


def _format_knowledge(kb_snippets: Optional[Iterable[str]]) -> str:
    if not kb_snippets:
        return ""
    lines = ["EXTERNAL KNOWLEDGE:"]
    for snippet in kb_snippets:
        lines.append(f"- {snippet}")
    LOGGER.debug("Adding %d KB snippets to prompt", len(lines) - 1)
    return "\n".join(lines) + "\n\n"


def _format_values() -> str:
    lines = ["Schwartz value definitions:"]
    for name in VALUE_NAMES:
        definition = VALUE_DEFINITIONS.get(name, "")
        if definition:
            lines.append(f"- {name}: {definition}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _format_instructions() -> str:
    return (
        "Instructions:\n"
        "- Output a comma-separated list of value names from the definitions above.\n"
        "- If no values are present, output: NONE\n"
        "- Output only the list (or NONE), no extra text."
    )


def _build_prompt(body: str, kb_snippets: Optional[Iterable[str]]) -> str:
    prompt = (
        f"{_format_knowledge(kb_snippets)}"
        f"{TASK_DESCRIPTION}\n\n"
        f"{_format_values()}\n\n"
        f"{_format_instructions()}\n\n"
        f"{body}"
    )
    LOGGER.debug("Built prompt with length %d", len(prompt))
    return prompt.strip()


def build_prompt_sentence(
    target_text: str, kb_snippets: Optional[Iterable[str]] = None
) -> str:
    body = f"TARGET SENTENCE:\n{target_text}"
    return _build_prompt(body, kb_snippets)


def build_prompt_window(
    context_text: str,
    target_text: str,
    kb_snippets: Optional[Iterable[str]] = None,
) -> str:
    body = (
        "CONTEXT WINDOW:\n" f"{context_text}\n\n" "TARGET SENTENCE:\n" f"{target_text}"
    )
    return _build_prompt(body, kb_snippets)


def build_prompt_doc(
    doc_text: str,
    target_text: str,
    kb_snippets: Optional[Iterable[str]] = None,
) -> str:
    body = "DOCUMENT:\n" f"{doc_text}\n\n" "TARGET SENTENCE:\n" f"{target_text}"
    return _build_prompt(body, kb_snippets)
