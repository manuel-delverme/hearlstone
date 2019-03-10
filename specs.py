import environments.simulator


def check_info_spec(info):
  assert sorted(info.keys()) == ['original_info', 'possible_actions']

  assert isinstance(info['original_info'], dict)
  assert isinstance(info['original_info']['possible_actions'], tuple)
  assert isinstance(info['original_info']['possible_actions'][0], environments.simulator.HSsimulation.Action)

  assert isinstance(info['possible_actions'], tuple)
  assert isinstance(info['possible_actions'][0], int)
