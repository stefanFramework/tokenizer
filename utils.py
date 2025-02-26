import re


def to_human_readable_name(obj) -> str:
    class_name = obj.__class__.__name__
    words = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name)
    return words.lower()
