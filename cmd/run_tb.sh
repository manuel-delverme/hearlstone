#!/usr/bin/env bash

source ~/hs/bin/activate

while true;
do
  #killall ngrok
  killall tensorboard
  ngrok http 6006
  tensorboard --logdir ~/d3sm0/logs
  tb_address=$(curl -s http://127.0.0.1:4040/api/tunnels | python -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")
  telegram-send "Tensorboard started at $1"
  sleep 20m
  bash ~/d3sm0/hearlstone/cmd/sync.sh
done
