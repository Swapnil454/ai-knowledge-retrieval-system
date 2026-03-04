from transformers import pipeline
from core.config import QA_MODEL

qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL
)


def generate_answer(question, context):

    result = qa_pipeline(
        question=question,
        context=context
    )

    return result["answer"]