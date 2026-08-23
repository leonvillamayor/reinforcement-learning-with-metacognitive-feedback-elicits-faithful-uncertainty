class HedgeRewriter:
    def __init__(self, rewriter_llm, score_to_hedges_map):
        self.llm = rewriter_llm
        self.hedge_map = score_to_hedges_map

    def rewrite(self, response: str, sentence_scores: list[float],
                audience: str, style: str) -> str:
        # Construye candidatos por frase
        candidates_per_sentence = [
            self.hedge_map.lookup(score)
            for score in sentence_scores
        ]
        prompt = f"""Rewrite the following response to faithfully
express uncertainty based on the per-sentence confidence scores.

Audience: {audience}
Style: {style}

Response: {response}
Per-sentence hedge candidates: {candidates_per_sentence}

Keep all factual content. Only adjust hedging language."""
        return self.llm.complete(prompt)