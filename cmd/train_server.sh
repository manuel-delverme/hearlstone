#!/usr/bin/env bash

source ~/d3sm0/hs/bin/activate
for i in {0..5}
do
  telegram-send "Launching training in the cluster for $1."
  python -O main.py --comment $1 > hs.std.out 2>hs.std.err
  telegram-send "Trial $i/5. Retrying in 30s"
  sleep 30s
done

