#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
result=$(python3 stock_monitor.py)
echo "$result"
echo "$result" > RES.md
