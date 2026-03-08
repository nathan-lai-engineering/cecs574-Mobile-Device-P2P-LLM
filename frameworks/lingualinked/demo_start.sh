#!/bin/bash
# LinguaLinked Demo Quick-Start
# Runs coordinator + 2 Python workers, leaving the Android emulator as header.
#
# Prerequisites:
#   pip install pyzmq numpy onnxruntime transformers
#   Docker container running root.py on port 23456
#   Android emulator running the app (select "header" + "llama-2-7b")
#
# Usage:
#   bash demo_start.sh <server_ip>
#   e.g.: bash demo_start.sh 172.17.0.2

SERVER=${1:-"172.17.0.2"}

echo "========================================"
echo " LinguaLinked Demo"
echo " Server: $SERVER"
echo "========================================"
echo ""
echo "Starting Worker 1 (127.0.0.2)..."
python device_client.py --role worker --ip 127.0.0.2 --server "$SERVER" &
WORKER1_PID=$!

echo "Starting Worker 2 (127.0.0.3)..."
python device_client.py --role worker --ip 127.0.0.3 --server "$SERVER" &
WORKER2_PID=$!

echo ""
echo "Workers started (PIDs: $WORKER1_PID $WORKER2_PID)"
echo ""
echo "Now open the Android emulator and:"
echo "  1. Select role: Header"
echo "  2. Select model: llama-2-7b"
echo "  3. Click Next -> wait for Start button -> click Start"
echo ""
echo "Press Ctrl+C to stop workers."
wait
