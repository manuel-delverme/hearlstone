#!/usr/bin/env bash

source ~/d3sm0/hs/bin/activate

while true;
do
  bash ~/d3sm0/hearlstone/cmd/sync.sh
  killall ngrok
  killall tensorboard
  /snap/ngrok/current/ngrok http 6006 > /dev/null & disown
  sleep 10
  echo "Starting tensorboard"
  tensorboard --logdir ~/d3sm0/logs &
  tb_address=$(curl -s http://127.0.0.1:4040/api/tunnels | python -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")
  telegram-send "Tensorboard started at $tb_address"
  sleep 60m
done
