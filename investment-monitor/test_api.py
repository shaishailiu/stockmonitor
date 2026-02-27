#!/usr/bin/env python3
"""
投资监控 - 简化测试版本
使用更稳定的数据获取方法
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import time

def test_stock_us(symbol):
    """测试美股数据获取"""
    print(f"\n测试美股: {symbol}")
    try:
        # 方法1: 使用 stock_us_spot_em 获取实时数据
        df = ak.stock_us_spot_em()
        print(f"  获取到 {len(df)} 条美股数据")
        print(f"  可用列: {df.columns.tolist()}")
        
        # 查找目标股票
        stock = df[df['代码'] == symbol]
        if not stock.empty:
            print(f"  ✅ 找到 {symbol}: 最新价 = {stock['最新价'].values[0]}")
        else:
            print(f"  ❌ 未找到 {symbol}")
            
    except Exception as e:
        print(f"  ❌ 错误: {e}")

def test_crypto():
    """测试加密货币数据"""
    print(f"\n测试加密货币: BTC")
    try:
        # 尝试 crypto_js_spot
        df = ak.crypto_js_spot()
        print(f"  获取到 {len(df)} 条加密货币数据")
        print(f"  可用列: {df.columns.tolist()}")
        print(f"  前5行:\n{df.head()}")
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")

def test_gold():
    """测试黄金数据"""
    print(f"\n测试黄金数据")
    try:
        # 方法1: 使用期货黄金
        df = ak.futures_main_sina(symbol="AU0", start_date="20250101", end_date="20260210")
        print(f"  获取到 {len(df)} 条黄金期货数据")
        if not df.empty:
            print(f"  最新价: {df.iloc[-1]['close']}")
            
    except Exception as e:
        print(f"  ❌ 错误: {e}")

def test_hk_stock():
    """测试港股"""
    print(f"\n测试港股: 0700 (腾讯)")
    try:
        df = ak.stock_hk_spot_em()
        print(f"  获取到 {len(df)} 条港股数据")
        print(f"  可用列: {df.columns.tolist()}")
        
        stock = df[df['代码'] == '00700']
        if not stock.empty:
            print(f"  ✅ 找到腾讯: 最新价 = {stock['最新价'].values[0]}")
            
    except Exception as e:
        print(f"  ❌ 错误: {e}")

if __name__ == "__main__":
    print("="*60)
    print("AkShare API 测试")
    print("="*60)
    
    # 测试各个数据源
    test_stock_us("AAPL")
    time.sleep(1)
    
    test_stock_us("NVDA")
    time.sleep(1)
    
    test_hk_stock()
    time.sleep(1)
    
    test_crypto()
    time.sleep(1)
    
    test_gold()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
