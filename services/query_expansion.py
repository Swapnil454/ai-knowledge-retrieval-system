from transformers import pipeline

# use supported task
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)


def generate_queries(question, num_queries=3):

    prompt = f"""
Generate {num_queries} search queries related to the question.

Question: {question}

Queries:
"""

    result = generator(
        prompt,
        max_length=64,
        num_return_sequences=1
    )

    text = result[0]["generated_text"]

    # split queries
    lines = text.split("\n")

    queries = []

    for line in lines:

        line = line.strip()

        if line and line != prompt.strip():

            queries.append(line)

    # ensure original query included
    queries.append(question)

    return queries