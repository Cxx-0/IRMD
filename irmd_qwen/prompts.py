from __future__ import annotations

from .data import Example, SequenceCorpus


SYSTEM_PROMPT = (
    "You are an expert sequential recommender. Follow the requested output format exactly."
)


def history_block(corpus: SequenceCorpus, example: Example, max_history: int) -> str:
    history = example.history[-max_history:]
    return "\n".join(
        f"{offset + 1}. [ITEM {item_id}] {corpus.describe(item_id)}"
        for offset, item_id in enumerate(history)
    )


def teacher_prompt(
    corpus: SequenceCorpus,
    example: Example,
    candidates: list[int],
    max_history: int,
) -> str:
    candidate_section = ""
    prediction_instruction = ""
    prediction_output = ""
    if candidates:
        candidate_text = "\n".join(
            f"- [ITEM {item_id}] {corpus.describe(item_id)}" for item_id in candidates
        )
        candidate_section = f"\nCandidate Items:\n{candidate_text}\n"
        prediction_instruction = "\n(6) Final Prediction: select exactly one candidate ID."
        prediction_output = "\n[PREDICTION]\n(A single integer candidate ID.)"
    return f"""Your task is to predict the user's next interaction. Please follow a strict Chain-of-Thought process.

User's Interaction History (oldest to newest):
{history_block(corpus, example, max_history)}
{candidate_section}

Instructions:
(1) Analyze Sequence: identify patterns, evolving interests, brand/category preferences.
(2) Infer User Intent: articulate the current need (exploring, upgrading, complementing, or changing focus).
(3) Reasoning: formulate step-by-step logic that leads to one candidate.
(4) Predict Attributes: describe the ideal attributes of the next item.
(5) Confidence Score: self-assess confidence from 0.0 to 1.0.{prediction_instruction}
(6) Brevity: keep reasoning under 120 words and always finish all requested output fields.

Output Format:
[REASONING]
(A concise step-by-step reasoning path that supports the inferred next-item attributes.)
[CONFIDENCE]
(A decimal number from 0.0 to 1.0.){prediction_output}"""


def grounded_teacher_prompt(
    corpus: SequenceCorpus,
    example: Example,
    max_history: int,
) -> str:
    """Generate training-only reasoning grounded in the observed next item."""
    return f"""Your task is to explain why the known next interaction is plausible from the user's prior history.

User's Interaction History (oldest to newest):
{history_block(corpus, example, max_history)}

Known Next Interaction (training supervision only):
[ITEM {example.target}] {corpus.describe(example.target)}

Instructions:
(1) Analyze only evidence present in the prior history and item descriptions.
(2) Connect stable preferences, recent intent, or complementary needs to the known next item.
(3) Do not claim facts that are absent from the supplied descriptions.
(4) Keep the reasoning under 120 words.
(5) Report a calibrated confidence from 0.0 to 1.0.
(6) Return the known next-item ID exactly so the record can be validated automatically.

Output Format:
[REASONING]
(A concise evidence-grounded reasoning path.)
[CONFIDENCE]
(A decimal number from 0.0 to 1.0.)
[PREDICTION]
{example.target}"""


def student_prompt(
    corpus: SequenceCorpus,
    example: Example,
    max_history: int,
    style: str = "irmd",
) -> str:
    history = example.history[-max_history:]
    if style == "catalog":
        sequence = "\n".join(
            f"{offset + 1}. <catalog_item_{item_id}> {corpus.describe(item_id)}"
            for offset, item_id in enumerate(history)
        )
        return f"""Your task is to predict the user's next interaction based on their interaction history.

User's Interaction History (oldest to newest):
{sequence}

Infer the user's intent internally. Return the single most likely next catalog item."""
    if style == "p5":
        sequence = " -> ".join(corpus.describe(item_id) for item_id in history)
        return (
            "A user interacted with the following products in chronological order:\n"
            f"{sequence}\n"
            "What product will the user most likely interact with next?"
        )
    if style == "p5_id":
        sequence = " -> ".join(str(item_id) for item_id in example.history)
        return (
            f"Here is the purchase history list of user_{example.user_id} : \n {sequence} \n "
            "try to recommend next item to the user"
        )
    if style == "bigrec":
        sequence = "\n".join(
            f"{offset + 1}. {corpus.describe(item_id)}"
            for offset, item_id in enumerate(history)
        )
        return (
            "Infer the user's current preference from this chronological product history:\n"
            f"{sequence}\n"
            "Represent the single most likely next product:"
        )
    if style == "tallrec":
        sequence = "; ".join(corpus.describe(item_id) for item_id in history)
        return (
            f"User history: {sequence}\n"
            "Question: Which catalog product is most likely to interest this user next?\n"
            "Answer:"
        )
    if style != "irmd":
        raise ValueError(f"unknown student prompt style: {style}")
    return f"""Your task is to predict the user's next interaction based on their interaction history.

User's Interaction History (oldest to newest):
{history_block(corpus, example, max_history)}

Instructions:
(1) Analyze Context internally: detect short-term triggers and long-term preferences.
(2) Infer Intent internally without outputting reasoning.
(3) Final Prediction: directly identify the item that best satisfies the inferred intent.

Output Format:
[PREDICTION]
(Predicted item ID or title.)"""
