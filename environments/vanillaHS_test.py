def HSenv_test():
  env = VanillaHS(skip_mulligan=True)
  s0, reward, terminal, info = env.reset()
  done = False
  step = 0
  for _ in range(3):
    while not done:
      step += 1
      possible_actions = info['possible_actions']
      random_act = random.choice(possible_actions)
      s, r, done, info = env.step(random_act)
      print(env.render(mode='ASCII'))
