#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
result=$(python3 stock_history_v2.py "$@")
echo "$result"
echo "$result" > data.md
