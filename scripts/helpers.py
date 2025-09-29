from rapidfuzz import process, fuzz

def fuzzy_get(d: dict, query: str, threshold: float = 90):
    """
    Return (best_key, value, score) of the most similar key in d to query
    if score >= threshold; else return (None, None, 0).
    """
    # extract one best match
    best = process.extractOne(
        query,                # what you’re looking for
        d.keys(),             # keys to search
        scorer=fuzz.ratio     # similarity metric
    )
    if best is None:
        return None, None, 0
    
    best_key, score, _ = best
    if score >= threshold:
        return best_key, d[best_key], score
    else:
        return None, None, score
