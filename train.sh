#!/usr/bin/env bash

# source ~/d3sm0/hs/bin/activate
while true
do
    echo "Starting the server..."
    cd ../; dotnet run -p SabberStone-grpc-testbed/SabberStone_gRPC >/dev/null &
    cd -
    sleep 10
    echo "Starting training"
    python -O main.py > logs/debug/std.out 2>logs/debug/std.err
    echo "Training eneded, committing files.."
    python bot.py
    killall dotnet
    sleep 10
done
