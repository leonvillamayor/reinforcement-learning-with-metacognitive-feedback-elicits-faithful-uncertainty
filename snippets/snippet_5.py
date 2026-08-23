# Esqueleto típico de implementación
def reward(completion, gold_answer):
    c_pred = parse_confidences(completion)          # [0,1] por oración
    c_gold = judge_implied_confidence(gold_answer)  # del judge externo
    r_faith = -((c_pred - c_gold) ** 2).mean()
    r_factual = 1 - (c_pred.mean() - is_correct(gold_answer)) ** 2
    r_acc = float(is_correct(gold_answer))
    r_format = strict_format_score(completion) + soft_format_score(completion)
    return (3*r_format + r_factual + r_acc + 12*r_faith)