#!/bin/bash
cd /home/ubuntu/MoneyGone

export ODDS_API_KEY=cc51d754710b03055256d945d8643c68

echo "Starting collector..."
nohup .venv/bin/python scripts/worker_collector.py \
    --config config/default.yaml --overlay config/stress-test.yaml \
    >> logs/collector-stress.log 2>&1 &
echo "Collector PID: $!"

echo "Starting execution engine..."
nohup .venv/bin/python scripts/run_live.py \
    --config config/paper.yaml --overlay config/stress-test.yaml \
    >> logs/stress-test-v8.log 2>&1 &
echo "Execution PID: $!"

echo "All processes started."
