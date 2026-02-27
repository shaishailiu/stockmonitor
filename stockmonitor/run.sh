#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
result=$(python3 stock_history.py)
echo "$result"
echo "$result" > data.md
