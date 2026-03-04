def chunk_text(pages, size=500, overlap=100):

    chunks = []

    for page in pages:

        words = page["text"].split()

        step = size - overlap

        for i in range(0, len(words), step):

            chunk_text = " ".join(words[i:i + size])

            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"]
            })

    return chunks