#!/bin/sh
# Copy src/ and config/ to /root/arm-app on the board, then
# (re)start the camera server there. Uses plain ssh/scp, so it asks for the
# board password unless a key or ControlMaster connection is set up.
set -eu

HOST="${BOARD_HOST:-root@192.168.137.100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/root/miniconda3/envs/ros-arm-env/bin/python

COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --exclude "__pycache__" --exclude ".DS_Store" -C "$ROOT" -cf - src config |
  ssh "$HOST" 'mkdir -p /root/arm-app && tar -C /root/arm-app -xf -'

# The pid file avoids pkill -f, whose pattern would also match this ssh command line.
ssh "$HOST" "cd /root/arm-app; [ -f camera_server.pid ] && kill \$(cat camera_server.pid) 2>/dev/null; sleep 1;
  nohup $PY src/adapters/board_camera_server.py --port 8080 > camera_server.log 2>&1 &
  echo \$! > camera_server.pid; sleep 2; curl -s http://127.0.0.1:8080/health"
echo
echo "camera: http://192.168.137.100:8080/"
