import re
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):

    return dot(a, b) / (norm(a) * norm(b) + 1e-8)


def split_sentences(text):

    if not text:
        return []

    sentences = re.split(r'(?<=[.!؟])\s+', text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 30
    ]


def rank_sentences_by_similarity(
    claim,
    sentences,
    embed_model,
    top_k=5
):

    if not sentences:
        return []

    claim_vector = embed_model.encode([claim])[0]

    sentence_vectors = embed_model.encode(sentences)

    scored_sentences = []

    for i, sentence in enumerate(sentences):

        score = cosine_similarity(
            claim_vector,
            sentence_vectors[i]
        )

        scored_sentences.append({
            "sentence": sentence,
            "score": float(score)
        })

    scored_sentences.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_sentences[:top_k]
