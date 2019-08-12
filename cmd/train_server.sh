#!/usr/bin/env bash

source ~/d3sm0/hs/bin/activate
killall dotnet
telegram-send "Starting server"
dotnet run -p ~/d3sm0/hearlstone/sb_env/SabberStone_gRPC rpc > sb.std.out 2>sb.std.err &
telegram-send "Sleeping 60s.."
sleep 60

for i in {0..5}
do
  telegram-send "Launching training in the cluster for $1."
  python -O main.py --comment $1 > hs.std.out 2>hs.std.err
  telegram-send "Trial $i/5. Retrying in 30s"
  sleep 30s
done

