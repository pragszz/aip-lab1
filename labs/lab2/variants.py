#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, apply_business_rules, extract_deterministic,
)
# Additional names needed for TicketRecordReasoned and the few-shot renderer.
from labs.lab1.extract import (  # noqa: E402
    CATEGORIES, CATEGORY_DESC, CONTAINS_PII_DESC, EVIDENCE_DESC, LANGUAGE_DESC,
    POLICY_NUMBER_DESC, PRODUCT_DESC, SENTIMENT_DESC, URGENCY_DESC,
)
from aip.llm import StructuredOutputError, structured  # noqa: E402

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# TODO A1: choose 6 dev-set tickets. For EACH, write one line saying what it
#          teaches that prose cannot. Pick edges, not averages (T2 §2.2):
#            - the billing/complaint boundary
#            - a ticket with no policy number (teaches null)
#            - a Hinglish ticket
#            - a satisfied-but-urgent ticket (the sentiment/urgency trap)
#            - a ticket whose policy number is only in a quoted reply
#            - one you got wrong in Lab 1
FEW_SHOT_IDS: list[str] = [
    "T0054",   # teaches: asks for "a full refund" but gold is complaint
    "T0021",   # teaches: Hinglish via transliteration alone, "Jaldi karo"
    "T0048",   # teaches: no AUR-<7 digit> string anywhere
    "T0222",   # teaches: 'satisfied' tone ("very good") sitting right next 
               # to a still-live, substantive question
    "T0020",   # teaches: Mention of two policy numbers
    "T0054",   # teaches: my own Lab 1 Part B miss, 4 of 8 graded fields
               # wrong at once (category).
]


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str]) -> str:
    """TODO A2: render the examples into the prompt.

    The example output format must be byte-identical to the format you are
    asking the model to produce. A mismatch here is a classic own goal.
    """
    examples = load_examples(ids)
    blocks = []
    for ex in examples:
        gold = ex["expected"]
        obj = {
            "evidence": ex["input"][:120],
            "category": gold["category"],
            "urgency": gold["urgency"],
            "sentiment": gold["sentiment"],
            "product": gold["product"],
            "language": gold["language"],
            "policy_number": gold["policy_number"],
            "contains_pii": gold["contains_pii"],
        }
        blocks.append(f"Ticket:\n{ex['input']}\n\nOutput:\n{json.dumps(obj)}")
    return "EXAMPLES:\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: Lab 1 Part C, no examples. This is your baseline."""
    try:
        rec = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier=tier)
        fields = rec.model_dump()
        needs_human_review = False
        review_reason = ""
    except StructuredOutputError as e:
        fields = {
            "evidence": "", "category": "information", "urgency": 1,
            "sentiment": "neutral", "product": "unknown", "language": "en",
            "policy_number": None, "contains_pii": False,
        }
        needs_human_review = True
        review_reason = f"structured output failed: {e}"

    fields = apply_business_rules(fields, ticket)
    fields.update(extract_deterministic(ticket))
    fields["needs_human_review"] = needs_human_review
    fields["review_reason"] = review_reason
    return fields


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: zero_shot + the few-shot block."""
    system = SYSTEM_PROMPT + "\n\n" + few_shot_block(FEW_SHOT_IDS)
    try:
        rec = structured(ticket, schema=TicketRecord, system=system, tier=tier)
        fields = rec.model_dump()
        needs_human_review = False
        review_reason = ""
    except StructuredOutputError as e:
        fields = {
            "evidence": "", "category": "information", "urgency": 1,
            "sentiment": "neutral", "product": "unknown", "language": "en",
            "policy_number": None, "contains_pii": False,
        }
        needs_human_review = True
        review_reason = f"structured output failed: {e}"

    fields = apply_business_rules(fields, ticket)
    fields.update(extract_deterministic(ticket))
    fields["needs_human_review"] = needs_human_review
    fields["review_reason"] = review_reason
    return fields


class TicketRecordReasoned(BaseModel):
    """TODO B: add a `reasoning: str` field FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition the
    answer; putting it last makes it a post-hoc rationalisation. You want the
    first. Measure the difference in output tokens.
    """

    reasoning: str = Field(
        description="Think through the ticket before deciding anything else."
    )
    evidence: str = Field(max_length=200, description=EVIDENCE_DESC)
    category: CATEGORIES = Field(description=CATEGORY_DESC)
    urgency: int = Field(ge=1, le=5, description=URGENCY_DESC)
    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description=SENTIMENT_DESC
    )
    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description=PRODUCT_DESC
    )
    language: Literal["en", "hi-en"] = Field(description=LANGUAGE_DESC)
    policy_number: str | None = Field(default=None, description=POLICY_NUMBER_DESC)
    contains_pii: bool = Field(default=False, description=CONTAINS_PII_DESC)
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if v == "" or v.lower() == "null":
            return None
        if not re.fullmatch(r"AUR-\d{7}", v):
            return None
        return v


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: few_shot with TicketRecordReasoned."""
    system = SYSTEM_PROMPT + "\n\n" + few_shot_block(FEW_SHOT_IDS)
    try:
        rec = structured(ticket, schema=TicketRecordReasoned, system=system, tier=tier)
        fields = rec.model_dump()
        needs_human_review = False
        review_reason = ""
    except StructuredOutputError as e:
        fields = {
            "reasoning": "", "evidence": "", "category": "information", "urgency": 1,
            "sentiment": "neutral", "product": "unknown", "language": "en",
            "policy_number": None, "contains_pii": False,
        }
        needs_human_review = True
        review_reason = f"structured output failed: {e}"

    fields = apply_business_rules(fields, ticket)
    fields.update(extract_deterministic(ticket))
    fields["needs_human_review"] = needs_human_review
    fields["review_reason"] = review_reason
    return fields


def cascade(ticket: str) -> dict:
    """TODO C: SMALL first; escalate to MAIN on a trigger you choose.

    Triggers, roughly in ascending order of how well they work:
      - validation failed                      (free, weak: misses confident errors)
      - evidence field empty or very short     (free, surprisingly decent)
      - urgency >= 4                           (free, but it is not a confidence signal)
      - two SMALL samples at T=0.7 disagree    (2x small cost, much the best)

    Record which path each ticket took -- set rec['_path'] = 'small' | 'large'
    so grid.py can report the escalation rate.
    """
    first = zero_shot(ticket, "SMALL")

    try:
        probe = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT,
                           tier="SMALL", temperature=0.7).model_dump()
    except StructuredOutputError:
        probe = None

    judgement_fields = ("category", "urgency", "sentiment", "product", "language")
    agree = probe is not None and all(
        first.get(f) == probe.get(f) for f in judgement_fields
    )

    if agree:
        first["_path"] = "small"
        return first

    large = zero_shot(ticket, "MAIN")
    large["_path"] = "large"
    return large


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
