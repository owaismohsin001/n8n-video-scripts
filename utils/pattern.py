from difflib import SequenceMatcher # returns np array frame


def text_similarity(a, b):
    a = a or ""
    b = b or ""
    print(f"Comparing texts:\nA: {a}\nB: {b}")
    total = len(a) + len(b)
    if total == 0:
        return 1.0  # nothing to compare
    matcher = SequenceMatcher(None, a, b)
    print("Matcher :", matcher)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    print(match)
    common = match.size
    print(common,"common")
    print(2 * common / total)
    return 2 * common / total