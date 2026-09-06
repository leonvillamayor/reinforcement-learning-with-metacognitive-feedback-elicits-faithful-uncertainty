import torch

def rlmf_advantage(rewards, faithfulness_idx=0, k=1.0, f_pred=None, f_gold=None):
    """
    rewards: [G, R_components]
    f_pred, f_gold: [G] precomputados (None para completions no escaladas)
    """
    rho = rewards.sum(dim=-1)
    rho_bar = rho.mean()
    A_raw = rho - rho_bar  # GRPO delta advantage

    f = rewards[:, faithfulness_idx]
    f_bar = f.mean()
    o = (rewards.sum(dim=-1) - f)  # resto de componentes
    o_bar = o.mean()

    # máscara: solo escalar completions "ganadoras"
    win_mask = (f > f_bar).float()

    # Z_g solo donde hay self-judge disponible
    Z = torch.where(
        torch.tensor([fg is not None for fg in f_gold]),
        1.0 - (torch.tensor(f_pred) - torch.tensor(f_gold))**2,
        torch.zeros_like(f)
    )

    faith_term = (f - f_bar) * (k + Z) * win_mask + (f - f_bar) * (1 - win_mask)
    A_rlmf = (o - o_bar) + faith_term
    return A_rlmf