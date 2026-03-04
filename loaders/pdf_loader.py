from pypdf import PdfReader


def load_pdf(file):

    file.seek(0)

    reader = PdfReader(file)

    pages = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            pages.append({
                "text": text,
                "page": i + 1,
                "source": file.name
            })

    return pages