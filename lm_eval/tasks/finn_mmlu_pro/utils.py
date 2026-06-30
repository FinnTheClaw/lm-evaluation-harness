import string


def doc_to_text(doc):
    doc_to_text = f"{doc['question']}\n"

    for i in range(len(doc["options"])):
        doc_to_text += f"{string.ascii_uppercase[i]}. {doc['options'][i]}\n"

    doc_to_text += "Answer:"
    return doc_to_text


def doc_to_choice(doc):
    return [string.ascii_uppercase[i] for i in range(len(doc["options"]))]


def doc_to_text_finn(doc):
    """generate_until variant with a direct answer-format instruction."""
    letters = string.ascii_uppercase
    n = len(doc["options"])
    text = (
        "The following is a multiple choice question. Choose the correct option "
        f"and respond with exactly: 'The best answer is X.' where X is one of {', '.join(letters[:n])}.\n\n"
        f"{doc['question']}\n"
    )
    for i in range(n):
        text += f"{letters[i]}. {doc['options'][i]}\n"
    text += "\nThe best answer is"
    return text
