# heaRLstone
```python
gm.set_oppoennt(RandomAgent())
player = load_ckpt()
rollouts = gather_rollouts(player)
player1 = ppo_update(rollouts)
save_model(player1)
wr = eval_vs_heuristic(player1)
old_wr = wr
player = player1

gm.add_learned_opponent(player1)

while 1:
  for ppo_iter in ..:
    player = load_ckpt(player)
    rollouts = gather_rollouts(player)
    player1 = ppo_update(rollouts)
    save_model(player1)

    if should_test():
      if eval_agent(player1) > 0.55:
        save_model(player1)
        break

  if old_wr < eval_vs_heuristic(player1):
    old_wr = wr
    player = player1
```
