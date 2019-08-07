#!/usr/bin/env bash

# source ~/d3sm0/hs/bin/activate
while true
do
    echo "Starting the server..."
   # dotnet run -p /home/tensorflow/d3sm0/SabberStone-grpc-testbed/SabberStone_gRPC > std.out 2>std.err &
#sleep 10
    echo "Starting training"
    telegram-send "Tensorboard can be seen here https://5dd7834f.ngrok.io/"
    python -O main.py --comment $1 > logs/debug/std.out 2>logs/debug/std.err
    echo "Training eneded, committing files.."
    python bot.py
    #killall dotnet
    sleep 10
done
