"""Step 6: prompt templates for grounded answers with inline citations.

The system prompt states the rules from doc 02 (answer only from the numbered passages, cite
them as [1] to [n], refuse with the exact sentence in ``REFUSAL_TEXT``). Two few-shot turns
show the behaviour: one answered with citations, one refused. Their passages come from a
made-up paper, so the examples teach the format without leaking facts into real answers.
When retrieval flags the context as weak, ``WEAK_CONTEXT_NOTICE`` is added to the question.
"""

from langchain_core.prompts import ChatPromptTemplate

from rpab.config import REFUSAL_TEXT
from rpab.schemas import RetrievedChunk

SYSTEM_PROMPT = f"""You are a research assistant answering questions about a fixed set of AI research papers.

Rules:
1. Answer ONLY using the numbered context passages in the user's message.
2. If the passages do not contain enough information to answer, reply exactly:
   "{REFUSAL_TEXT}"
   and nothing else.
3. Cite sources inline as [1], [2], [3], matching the numbered passages. Every sentence that states
   a fact from a passage ends with its citation. Never cite a number that is not in the context.
4. Do not use outside knowledge. Do not invent paper titles, page numbers or results.
5. Be concise and precise. Use technical terms as the papers use them."""

USER_TEMPLATE = """Context:
{context}
{notice}
Question: {question}"""

# Added to the user message when the top passage scores below the calibrated weak-context threshold.
# Doc 03 still lets the model answer then; the notice only makes refusing the expected default.
WEAK_CONTEXT_NOTICE = ("\nNote: these passages scored low for this question, so they may not contain the answer. "
                       "Answer only if a passage states it directly; otherwise reply with the refusal sentence "
                       "exactly.\n")

# Few-shot passages from a fictional paper: the format is real, the content is not
_EXAMPLE_CONTEXT = """[1] Paper: Sparse Gating for Efficient Sequence Models | Page: 4
We train all models with the Adam optimizer (beta1 = 0.9, beta2 = 0.98) and a peak learning rate of
3e-4, warmed up linearly over the first 2,000 steps and then decayed with a cosine schedule.

[2] Paper: Sparse Gating for Efficient Sequence Models | Page: 6
Table 2 reports perplexity on the validation set. The gated model with 8 experts reaches 18.2,
against 19.6 for the dense baseline of the same compute budget.

[3] Paper: Sparse Gating for Efficient Sequence Models | Page: 2
Earlier sparse models routed each token to a single expert, which made training unstable."""

FEW_SHOT = [
    ("human", USER_TEMPLATE.format(
        context=_EXAMPLE_CONTEXT, notice="",
        question="Which optimizer and learning rate schedule are used, and how does the gated model compare "
                 "with the dense baseline?")),
    ("ai", "The models are trained with Adam (beta1 = 0.9, beta2 = 0.98) at a peak learning rate of 3e-4, "
           "with a linear warm-up over the first 2,000 steps followed by cosine decay [1]. With 8 experts the "
           "gated model reaches a validation perplexity of 18.2, against 19.6 for the dense baseline of the "
           "same compute budget [2]."),
    # The refusal example carries the notice, as a weak-context question would
    ("human", USER_TEMPLATE.format(context=_EXAMPLE_CONTEXT, notice=WEAK_CONTEXT_NOTICE,
                                   question="How many GPUs were used to train the gated model?")),
    ("ai", REFUSAL_TEXT),
]

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT)]
    # Literal braces in the examples must not be read as template variables
    + [(role, text.replace("{", "{{").replace("}", "}}")) for role, text in FEW_SHOT]
    + [("human", USER_TEMPLATE)]
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the passages 1 to n with their paper title and page, as the prompt cites them."""
    return "\n\n".join(f"[{rank}] Paper: {c.paper_title} | Page: {c.page}\n{c.text.strip()}"
                       for rank, c in enumerate(chunks, start=1))
