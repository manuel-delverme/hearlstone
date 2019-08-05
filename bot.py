#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import datetime
import hs_config
import shutil

def save_and_commit():
  link = "https://github.com/d3sm0/hrl_logs/tree/master/debug"
  try:
    ckpt = sorted(glob.glob(os.path.join(hs_config.PPOAgent.save_dir,  '*[Ss]abberstone*')), key=os.path.getctime)[-1]
    print(ckpt)
    shutil.copy(ckpt, os.path.join(os.path.dirname(ckpt), 'latest.pt'))
  except IndexError:
    pass
  msg = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
  os.system(f"cd {hs_config.PPOAgent.log_dir}; git add . ; git commit -m {msg}; git push origin master")
  out = os.popen("git rev-parse HEAD")
  os.system(f"telegram-send 'latest commit {out.read().strip()}, dated at {msg}'")
  os.system(f"telegram-send 'Check for {link}'")


if __name__ == "__main__":
  save_and_commit()

