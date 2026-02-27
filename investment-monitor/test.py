#!/usr/bin/env python3
"""
快速测试脚本 - 只扫描少量资产
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from monitor import InvestmentMonitor, ASSETS

# 测试用的少量资产
TEST_ASSETS = {
    "us_tech": [
        {"symbol": "AAPL", "name": "苹果"},
        {"symbol": "NVDA", "name": "英伟达"}
    ],
    "crypto": [
        {"symbol": "BTC", "name": "比特币", "type": "crypto"}
    ]
}

# 临时替换
original_assets = ASSETS.copy()
ASSETS.clear()
ASSETS.update(TEST_ASSETS)

monitor = InvestmentMonitor()
print("🧪 测试模式：只扫描 AAPL, NVDA, BTC\n")
monitor.run_daily_scan()

# 恢复
ASSETS.clear()
ASSETS.update(original_assets)
