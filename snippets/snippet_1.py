def metacognitive_advantage(group_completions, grader):
    # group_completions: list of (response, parsed_conf, gold_reward, faith_reward)
    rewards_faith  = [c.faith_reward  for c in group_completions]
    rewards_other  = [c.gold_reward - c.faith_reward for c in group_completions]
    A_faith = standardize(rewards_faith)         # GRPO baseline sobre faith
    A_rest  = standardize(rewards_other)         # GRPO baseline sobre el resto
    f_g     = [c.faith_reward for c in group_completions]
    f_bar   = mean(f_g)
    # Z_g: qué tan bien predijo el modelo su propio F-score
    Z_g     = [(grader.F_pred(c) - grader.F_gold(c)) / std_F for c in group_completions]
    out = []
    for a_f, a_r, fg, z in zip(A_faith, A_rest, f_g, Z_g):
        if fg > f_bar:
            out.append(a_f * (1.0 + z) + a_r)
        else:
            out.append(a_f + a_r)
    return out