#!/usr/bin/env bash

for i in {1...5}
do
  killall dotnet
  telegram-send "Starting the server..."
  dotnet run -p ~/d3sm0/hearlstone/sb_env/SabberStone_gRPC $1 > sb.std.out 2>sb.std.err &
  sleep 30s
  telegram-send "Launching training in the cluster for $1. Connecting at $2"
  python -O main.py --comment $1 > hs.std.out 2>hs.std.err
  telegram-send "Launch failed. Stderr follows..."
  cat hs.std.err | telegram-send --stdin
  telegram-send "Trial $i/5. Retrying in 30s"
  sleep 30s
done

