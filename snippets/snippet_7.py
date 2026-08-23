# Pseudo-código de lo que el paper implica para un pipeline
class UncertaintyGate:
    def __init__(self, llm, context: Literal["casual", "scientific", "high_stakes"]):
        self.system_prompt = CONTEXT_PROMPTS[context]  # Figura 28
        self.min_diversity = check_hedge_diversity       # Criterio 1
        self.helpfulness_check = calibrate_user          # Criterio 3
    
    def generate(self, query: str) -> Response:
        # El RL con metacognitive feedback (paper) entrena al modelo
        # para que <sentence><confidence>...</confidence></sentence>
        # esté alineado con la verdadera incertidumbre epistémica
        return self.llm.generate(
            system=self.system_prompt,
            structured_output=("sentence", "confidence")
        )
    
    def evaluate_human(self, response_a, response_b, context):
        # Las instrucciones de la Figura 29 son el gold standard
        # de rúbrico: 4 ejes, anti-bias rules explícitas
        return annotate_preference(response_a, response_b, 
                                   dimensions=["diversity", "naturalness", 
                                               "helpfulness", "context"])