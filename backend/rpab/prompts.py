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


# ---- follow-up rewriting (stretch S1) ----------------------------------------------------

CONDENSE_SYSTEM = """You rewrite the user's latest question from a conversation about AI research papers into a
standalone question that can be understood without the conversation.

Rules:
1. A question that leans on the conversation in any way is not standalone: a pronoun ("it", "they",
   "this"), a reference such as "that paper" or "the second method", or a fragment such as
   "And the large one?". Rewrite it with the names and the topic it refers to.
2. Never leave "they" or "the authors" in the question: write the name of the paper or method instead.
3. A reference points to the most recent exchange unless the user names an earlier topic.
4. Keep the user's wording otherwise. Do not answer the question and do not add details.
5. Only a question that is complete on its own and starts a new topic is returned unchanged.
6. Reply with the question only, on one line."""

CONDENSE_TEMPLATE = """Conversation:
{history}

Latest question: {question}

Standalone question:"""

# Examples from the same fictional paper as the answer prompt's, so no real facts leak into rewrites
_CONDENSE_HISTORY = """User: How many experts does the Sparse Gating model use?
Assistant: The base Sparse Gating model routes each token to 2 of 8 experts."""

# Two topics: the reference belongs to the later one
_CONDENSE_HISTORY_TWO_TOPICS = _CONDENSE_HISTORY + """
User: What vocabulary size does the Sparse Gating tokenizer use?
Assistant: The base tokenizer has a vocabulary of 32,000 tokens."""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM),
    ("human", CONDENSE_TEMPLATE.format(history=_CONDENSE_HISTORY, question="What learning rate did they use?")),
    ("ai", "What learning rate did the Sparse Gating paper use?"),
    ("human", CONDENSE_TEMPLATE.format(history=_CONDENSE_HISTORY, question="And the large one?")),
    ("ai", "How many experts does the large Sparse Gating model use?"),
    ("human", CONDENSE_TEMPLATE.format(history=_CONDENSE_HISTORY_TWO_TOPICS, question="And the large one?")),
    ("ai", "What vocabulary size does the large Sparse Gating tokenizer use?"),
    ("human", CONDENSE_TEMPLATE.format(history=_CONDENSE_HISTORY,
                                       question="What optimizer does the Transformer paper use?")),
    ("ai", "What optimizer does the Transformer paper use?"),
    ("human", CONDENSE_TEMPLATE),
])
