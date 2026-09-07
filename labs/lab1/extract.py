#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]

EVIDENCE_DESC = (
    "Before choosing anything else, quote the exact span of the ticket "
    "(verbatim, <=200 characters, no paraphrasing, no ellipses) that most "
    "directly justifies the category you are about to assign. Copy the "
    "characters exactly as they appear, including any typos."
)
 
CATEGORY_DESC = (
    "Exactly one of: "
    "'billing' - questions or problems about premiums, invoices, payments, "
    "refunds, or charges; "
    "'claims' - reporting, tracking, or disputing an insurance claim "
    "(accidents, damage, payout amount, claim status); "
    "'policy_change' - a request to modify an existing policy: add/remove "
    "coverage, change a beneficiary, renew, upgrade/downgrade tier, or "
    "cancel; "
    "'technical' - trouble using the website, app, portal, or login, "
    "unrelated to the content of a claim or bill; "
    "'complaint' - the ticket's primary content is dissatisfaction "
    "(rude staff, long wait, being ignored) with NO specific actionable "
    "request attached; "
    "'information' - a general question not tied to the customer's own "
    "policy, claim, or bill (e.g. asking what a coverage type means). "
    "Boundary rule: if the ticket both expresses frustration AND asks for "
    "a specific fix (a refund, a claim to be reprocessed, a policy to be "
    "changed), classify by the fix being requested, not as 'complaint'. "
    "Reserve 'complaint' for tickets with no actionable request."
)
 
URGENCY_DESC = (
    "Integer 1-5 rating how quickly this ticket needs a human response. "
    "First ask: can this be answered without opening the customer's "
    "account or record? If yes, it is 1 -- purely informational or a "
    "self-service how-to, however long the message is (e.g. 'what does "
    "comprehensive cover mean?', 'where do I download my e-card?'). If "
    "Aurora must look up, act on, or fix something for this customer, or a "
    "transaction is in flight, it is at least 2 (e.g. 'how many wellness "
    "points do I have?', 'please add my newborn', 'the app crashes on "
    "upload'). "
    "3 = something has already gone wrong or is stuck and the customer is "
    "waiting (e.g. 'debited twice', a portability request gone quiet). "
    "4 = repeated failure to resolve, money or access at risk right now, "
    "or the customer THREATENS an escalation without yet taking it (e.g. "
    "'THIS IS THE THIRD TIME', 'refund it or I am going to the "
    "ombudsman'). "
    "5 = an emergency in progress, a formal denial demanding immediate "
    "reversal, or the customer STATES they already are escalating to the "
    "Ombudsman (e.g. 'father is in ICU and cashless is DENIED', 'I am "
    "filing a complaint with the ombudsman') -- note this is a tense "
    "distinction from 4, not a topic distinction: merely threatening the "
    "ombudsman is 4, stating you already are is 5. "
    "Modifier: add 1 (capped at 5) if the message states a same-day or "
    "next-morning deadline. "
    "Judge the situation, not the volume: shouting is a sentiment signal, "
    "not an urgency signal -- a calm ICU message is 5, a furious tax-"
    "certificate message is 2. Use 2 and 4 for cases that sit between two "
    "anchors rather than defaulting to 3."
)

SENTIMENT_DESC = (
    "The customer's emotional tone in THIS message, tone only, "
    "independent of urgency, one of: 'angry' (hostile language, all-caps, "
    "explicit blame, threats), 'frustrated' (unhappy and tired of trying, "
    "still civil -- REQUIRES the message to reference a prior failure: a "
    "repeat attempt, an unanswered request, a delay, or something not "
    "working), 'neutral' (matter-of-fact, no emotional language either "
    "way -- a first-time request is neutral, however terse, even if "
    "urgent), 'satisfied' (thanking, praising, or confirming something "
    "worked)."
)
 
PRODUCT_DESC = (
    "The policy tier the ticket concerns: 'bronze', 'silver', 'gold', "
    "'platinum', or 'unknown'. Use 'unknown' whenever the tier is not "
    "explicitly named or cannot be inferred with confidence, do not "
    "guess from price or coverage details mentioned."
)
 
LANGUAGE_DESC = (
    "'en' if the ticket is written in English only. 'hi-en' if the ticket "
    "code-mixes Hindi and English in the same sentence or message "
    "(Hinglish), even if only a few words are Hindi -- this includes "
    "Hindi written in Devanagari script AND Hindi transliterated into "
    "Latin script (e.g. 'kripya', 'jaldi', 'bahut', 'turant'). Otherwise "
    "'en'."
)

POLICY_NUMBER_DESC = (
    "The customer's policy number EXACTLY as it appears in the ticket, in "
    "the format AUR-<7 digits> (e.g. AUR-1234567). Copy it character for "
    "character -- do not reformat, pad, or correct it. Only take it from "
    "the CURRENT, live message. Lines beginning with '>' are a quoted "
    "reply from an earlier thread and may carry a different, stale "
    "policy number -- ignore policy numbers that appear only inside "
    "quoted lines or a signature block. If no policy number of this form "
    "appears in the live message, you MUST return null. Never invent a "
    "policy number and never infer one from other details."
)

CONTAINS_PII_DESC = (
    "True if the ticket contains a phone number, or an email address "
    "that is NOT one of Aurora's own published addresses "
    "(support@aurorahealth.example, grievance@aurorahealth.example). A "
    "personal name alone does not count as PII for this field."
)


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else.

    Remember from T2 §3.2: field `description`s are shipped to the model as
    part of the JSON Schema. They are the highest-leverage place to put an
    instruction, because they sit next to the thing they govern. Write them as
    instructions to the model, not as documentation for a human.
    """

    # B1a decision: evidence stays FIRST (before category/urgency/etc). This
    # is the WithReasoning idiom from T2 §3.3 -- evidence conditions the
    # fields that follow it, instead of just post-hoc justifying them, at
    # the cost of a few extra output tokens per record.
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

    # Part B only: the model decides these. In Part C you will delete them
    # from this schema and compute them in code instead.
    policy_number: str | None = Field(
        default=None,
        description=POLICY_NUMBER_DESC
    )

    contains_pii: bool = Field(
        default=False,
        description=CONTAINS_PII_DESC
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        # B1j decision: coerce to None rather than raise on "", "null", or a
        # non-conforming string. A raise here would burn a repair round-trip
        # on a field that is not safety-critical -- silently downgrading a
        # bad guess to null is cheap and matches the "never invent, use null
        # when absent" contract already given to the model.
        if v is None:
            return None
        v = v.strip()
        if v == "" or v.lower() == "null":
            return None
        if not re.fullmatch(r"AUR-\d{7}", v):
            return None
        return v


SYSTEM_PROMPT = """\
You are a triage classifier for Aurora Insurance's support inbox.
 
TASK
Read one customer support ticket and extract the fields defined by the
JSON schema you are given. Output nothing but a single JSON object matching
that schema -- no prose, no markdown fences, no keys beyond the schema.
 
CONTEXT
Tickets come from Aurora's own customers, in English or Hindi-English
code-mixed text, and may include a quoted earlier message below the current
one, marked with a leading '>' on each quoted line.
 
CONSTRAINTS
- Base every field only on what the ticket text says. Do not use outside
  knowledge about the customer, the company, or typical insurance cases.
- If a fact is not stated in the ticket, use the schema's null/"unknown"
  value for that field. Never guess to avoid leaving a field empty.
- Quoted-history text (lines starting with '>') is context, not the current
  request -- weight it accordingly.
 
Field-by-field instructions are attached to each field in the schema; follow
those exactly, they take precedence over anything general stated above.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    try:
        return structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT)
    except StructuredOutputError as e:
        # Must never raise: fall back to a sentinel record and flag it for a
        # human. model_construct bypasses validation, which is fine here
        # since every value below is a deliberately-chosen placeholder, not
        # untrusted model output.
        return TicketRecord.model_construct(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=f"structured output failed: {e}",
        )

# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


_AURORA_EMAILS = {"support@aurorahealth.example", "grievance@aurorahealth.example"}


def extract_deterministic(ticket: str) -> dict:
    """Return {'policy_number', 'contains_pii'} without a model call.

    policy_number:
        Find AUR-<7 digits> in the LIVE message only.

        C3 -- the trap. Some tickets could contain TWO policy-number-shaped
        strings: one in the live body, one in a quoted reply below a '>'
        line from an earlier thread, and they are not guaranteed to be the
        same number (data/README.md: "A ticket whose only policy-shaped
        string is inside a quoted reply is labelled null"). The rule: strip
        every line that starts with '>' (after stripping leading
        whitespace) before searching, so a stale number in quoted history
        can never win over -- or stand in for -- a missing live one.

        Generalises or fitted? Checked against the ticket generator's own
        rule (data/README.md), so the rule itself is dataset-independent.
        But empirically it never fires here: 0/240 tickets in this corpus
        (dev+test+blind) contain an AUR-<7 digits> string inside a quoted
        '>' line at all, so this branch is completely unexercised by our
        eval numbers. It is defensive code for a case this benchmark
        cannot reward or punish -- worth keeping for a production inbox
        where forwarded threads are common, worth flagging as untested here.

    contains_pii:
        True if the ticket contains a phone number, or an email address
        that is not one of Aurora's own published addresses. Verified
        against all 180 labelled dev+test tickets: 0 mismatches. A bare
        name is deliberately excluded (data/README.md), which the page
        itself says is too narrow for DPDP-Act purposes -- fine for a
        cost-reduction triage flag, not sufficient as a compliance control.
    """
    live_text = "\n".join(
        line for line in ticket.splitlines() if not line.strip().startswith(">")
    )
    match = POLICY_RE.search(live_text)
    policy_number = match.group(0) if match else None

    phone_hit = bool(_PII_PATTERNS["PHONE_IN"].search(ticket))
    email_hit = any(
        e.lower() not in _AURORA_EMAILS
        for e in _PII_PATTERNS["EMAIL"].findall(ticket)
    )

    return {"policy_number": policy_number, "contains_pii": phone_hit or email_hit}


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """Compute `escalate` in code -- a business rule, not a model judgement.

    escalate = urgency >= 4 or 'ombudsman' appears in the ticket

    Lives in code so a compliance officer can read it, change it without
    touching a prompt, and it can be unit-tested (see tests/test_extract_c.py).
    """
    escalate = rec_fields["urgency"] >= 4 or "ombudsman" in ticket.lower()
    return {**rec_fields, "escalate": escalate}


class TicketRecordC(BaseModel):
    """The reduced schema the model sees in Part C.

    `policy_number` and `contains_pii` are gone -- computed deterministically
    in `extract_deterministic`. `escalate` never existed on the model's side;
    it is a pure business rule from `apply_business_rules`. Fewer fields means
    a shorter prompt, fewer output tokens, and three fields locked at 100%
    accuracy instead of merely measured at it.
    """

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


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else.

    Returns a plain dict (model fields + deterministic fields + business rules)
    so that run_eval.py can score it against the gold labels directly.
    """
    try:
        rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT)
        fields = rec.model_dump()
        needs_human_review = False
        review_reason = ""
    except StructuredOutputError as e:
        # Same contract as extract_b: never raise, flag for a human instead.
        fields = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
        }
        needs_human_review = True
        review_reason = f"structured output failed: {e}"

    fields = apply_business_rules(fields, ticket)
    fields.update(extract_deterministic(ticket))
    fields["needs_human_review"] = needs_human_review
    fields["review_reason"] = review_reason
    return fields


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))