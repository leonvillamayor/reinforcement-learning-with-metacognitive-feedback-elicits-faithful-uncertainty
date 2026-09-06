# utilidad = "dónde me equivoco / no sé" − ruido OOD
def metacognitive_utility(x, y_star):
    F_pred = self_judge(x, y_star)         # 0-1, ya en no_grad
    c      = intrinsic_confidence(x, y_star)  # cons. entre G rollouts
    ood    = ood_detector(x)                  # 0-1, p.ej. Mahalanobis sobre hidden
    return alpha * (1 - c) + beta * (1 - F_pred) - gamma * ood