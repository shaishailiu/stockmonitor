#!/usr/bin/env python3
"""测试所有市场（A股、港股、美股）"""
from data_fetcher_v2 import HybridDataFetcher, IndicatorCalculator

fetcher = HybridDataFetcher()
calc = IndicatorCalculator()

print("="*60)
print("全市场数据源测试")
print("="*60)

test_cases = [
    ('sh600519', 'A', '贵州茅台'),
    ('hk00700', 'HK', '腾讯'),
    ('hk03690', 'HK', '美团'),
    ('NVDA', 'US', '英伟达'),
]

for symbol, market, name in test_cases:
    print(f"\n{'='*60}")
    print(f"{name} ({symbol}-{market})")
    print(f"{'='*60}")
    
    result = fetcher.fetch_stock_data(symbol, market)
    
    if result['success']:
        drawdown = calc.calculate_drawdown(result['current_price'], result['high_52w'])
        
        # 计算RSI
        rsi = None
        if len(result['prices']) >= 15:
            closes = [p['close'] for p in result['prices']]
            rsi = calc.calculate_rsi(closes, 14)
        
        print(f"✅ 数据源: {result['source']}")
        print(f"💰 当前价: {result['current_price']:.2f}")
        print(f"📈 52周高: {result['high_52w']:.2f} ({result['high_date']})")
        print(f"📉 52周低: {result['low_52w']:.2f}")
        print(f"📊 回撤: {drawdown:.2f}%")
        print(f"📅 数据: {result['days']} 天")
        if rsi:
            print(f"🔍 RSI(14): {rsi:.2f}")
        
        # 预警判断
        if drawdown >= 50:
            print(f"🔴 红色预警区域")
        elif drawdown >= 40:
            print(f"🟠 橙色预警区域")
        elif drawdown >= 30:
            print(f"🟡 黄色预警区域")
        else:
            print(f"✅ 正常区域")
    else:
        print(f"❌ {result['error']}")

print("\n" + "="*60)
print("全市场测试完成")
print("="*60)
