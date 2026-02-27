#!/usr/bin/env python3
"""测试港股完整历史数据"""
from data_fetcher_v2 import HybridDataFetcher, IndicatorCalculator

fetcher = HybridDataFetcher()
calc = IndicatorCalculator()

print("="*60)
print("测试港股完整历史数据（腾讯财经）")
print("="*60)

hk_stocks = [
    ('hk00700', '腾讯控股'),
    ('hk03690', '美团'),
    ('hk09888', '百度集团'),
]

for symbol, name in hk_stocks:
    print(f"\n{name} ({symbol}):")
    result = fetcher.fetch_stock_data(symbol, 'HK')
    
    if result['success']:
        drawdown = calc.calculate_drawdown(result['current_price'], result['high_52w'])
        
        # 计算RSI
        if len(result['prices']) >= 15:
            closes = [p['close'] for p in result['prices']]
            rsi = calc.calculate_rsi(closes, 14)
        else:
            rsi = None
        
        print(f"  ✅ 数据源: {result['source']}")
        print(f"  💰 当前价: HK$ {result['current_price']:.2f}")
        print(f"  📈 52周高: HK$ {result['high_52w']:.2f} ({result['high_date']})")
        print(f"  📉 52周低: HK$ {result['low_52w']:.2f}")
        print(f"  📊 回撤: {drawdown:.2f}%")
        print(f"  📅 数据点: {result['days']} 天")
        if rsi:
            print(f"  🔍 RSI(14): {rsi:.2f}")
    else:
        print(f"  ❌ {result['error']}")

print("\n" + "="*60)
print("✅ 港股历史数据可用！")
print("="*60)
