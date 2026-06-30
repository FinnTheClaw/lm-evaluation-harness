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
    n = len(choices_list)
    valid_letters = ", ".join(LETTERS[:n])
    choices_str = ""
    for i, choice in enumerate(choices_list):
        choices_str += f"{LETTERS[i]}. {choice}\n"

    text = (
        f"{doc['narrative']}\n\n"
        f"{doc['question']}\n\n"
        f"{choices_str}\n"
        f"Choose the best option and end your response with exactly: 'The best answer is X.' "
        f"where X is one of {valid_letters}.\n\nThe best answer is"
    )
    return text


def doc_to_target_finn(doc):
    """
    Return the letter (A/B/C...) corresponding to the correct answer,
    matching what doc_to_text_finn asks the model to output.
    """
    choices_list = ast.literal_eval(doc["choices"])
    answer_choice = doc.get("answer_choice", doc.get("target", ""))
    try:
        idx = choices_list.index(answer_choice)
        return LETTERS[idx]
    except (ValueError, IndexError):
        # fallback: use answer_index if available
        idx = doc.get("answer_index", 0)
        return LETTERS[idx]
