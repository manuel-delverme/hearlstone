#!/usr/bin/env bash

source ~/hs/bin/activate
telegram-send "Syncing from server..."
rsync -az -P ~/d3sm0/hearlstone/logs ~/d3sm0/logs/server/
telegram-send "Syncing from cluster..."
rsync -az -P cluster:~/hearlstone/logs ~/d3sm0/logs/cluster/
telegram-send "Sending to https://drive.google.com/open?id=1BH1UDG68ZR6j0nX9r9iz64tV9_lDPIE5"
rclone move -P ~/d3sm0/logs drive:HSCheckpoints


