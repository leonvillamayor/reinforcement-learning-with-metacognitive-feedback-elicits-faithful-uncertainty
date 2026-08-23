# Original (elegida)
A_RLMF_g = A_g * (k + Z_g)   if f_g > f else A_g

# Variante 1: escalar TODO el advantage
A_RLMF_g = A_g * (k + Z_g)   siempre

# Variante 2: aplicar a todo el grupo
A_RLMF_g = (o_g - o) + (f_g - f) * (k + Z_g)