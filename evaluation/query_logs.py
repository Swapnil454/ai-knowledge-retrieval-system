import json
from datetime import datetime


def log_query(question, answer):

    data = {
        "question": question,
        "answer": answer,
        "timestamp": str(datetime.now())
    }

    with open("data/query_logs.json", "a") as f:

        f.write(json.dumps(data) + "\n")