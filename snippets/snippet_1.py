# Pseudocódigo simplificado del filtro ARLMF
def arlmf_filter(M, D_train, N_target):
    # 1. Generar respuestas + hedges linguísticos
    responses, hedges = M.generate_with_uncertainty(D_train)

    # 2. Self-score: ¿mi confianza verbal coincide con mi confianza interna?
    #    Escala 0–100. g_i = |verbal_score - internal_conf|
    g_scores = M.meta_score_alignment(responses, hedges)

    # 3. Selección por extremos (estratificada)
    keep_low  = top_k_by(g_scores, k=N_target//2, ascending=True)   # no sabe y lo dice
    keep_high = top_k_by(g_scores, k=N_target//2, ascending=False)  # sabe y calla o al revés
    return D_train[keep_low + keep_high]