import ast


def doc_to_choice(doc):
    """
    Convert a doc to a choice.
    """
    return ast.literal_eval(doc["choices"])


DOC_TO_TEXT = "{narrative}\n\n{question}\n\n{choices}\nAnswer:"


def doc_to_text(doc):
    """
    Convert a doc to text.
    """
    choices = ""
    for i, choice in enumerate(ast.literal_eval(doc["choices"])):
        choices += f"{i + 1} - {choice}\n"

    text = DOC_TO_TEXT.format(
        narrative=doc["narrative"], question=doc["question"], choices=choices
    )

    return text


LETTERS = "ABCDEFGHIJ"


def doc_to_text_finn(doc):
    """
    generate_until variant: uses A/B/C/D labels instead of 1/2/3.
    Appends instruction to conclude with 'The best answer is X.'
    """
    choices_list = ast.literal_eval(doc["choices"])
    choices_str = ""
    for i, choice in enumerate(choices_list):
        choices_str += f"{LETTERS[i]}. {choice}\n"

    text = (
        f"{doc['narrative']}\n\n"
        f"{doc['question']}\n\n"
        f"{choices_str}\n"
        "Think through the problem carefully, then conclude with exactly: "
        "'The best answer is X.' where X is A, B, C, or D.\nAnswer:"
    )
    return text
