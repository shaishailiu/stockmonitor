#!/usr/bin/env python3
"""测试美股数据"""
from data_fetcher_v2 import HybridDataFetcher, IndicatorCalculator

fetcher = HybridDataFetcher()
calc = IndicatorCalculator()

print("="*60)
print("测试美股数据（Alpha Vantage）")
print("="*60)

us_stocks = [
    ('AAPL', '苹果'),
    ('NVDA', '英伟达'),
    ('TSLA', '特斯拉'),
]

for symbol, name in us_stocks:
    print(f"\n{name} ({symbol}):")
    result = fetcher.fetch_stock_data(symbol, 'US')
    
    if result['success']:
        drawdown = calc.calculate_drawdown(result['current_price'], result['high_52w'])
        print(f"  ✅ 当前价: ${result['current_price']:.2f}")
        print(f"  📈 52周高: ${result['high_52w']:.2f} ({result['high_date']})")
        print(f"  📉 回撤: {drawdown:.2f}%")
        print(f"  📊 数据点: {result['days']} 天")
    else:
        print(f"  ❌ {result['error']}")

print("\n" + "="*60)
