#!/usr/bin/env python3
"""检查小米的详细数据"""
from data_fetcher_v2 import HybridDataFetcher, IndicatorCalculator

fetcher = HybridDataFetcher()
calc = IndicatorCalculator()

print("="*60)
print("小米集团（01810.HK）详细分析")
print("="*60)

result = fetcher.fetch_stock_data('hk01810', 'HK')

if result['success']:
    drawdown = calc.calculate_drawdown(result['current_price'], result['high_52w'])
    
    print(f"\n💰 价格信息：")
    print(f"  当前价：HK$ {result['current_price']:.2f}")
    print(f"  52周最高：HK$ {result['high_52w']:.2f} ({result['high_date']})")
    print(f"  52周最低：HK$ {result['low_52w']:.2f}")
    print(f"  回撤幅度：{drawdown:.2f}%")
    print(f"  数据天数：{result['days']} 天")
    
    # 计算RSI
    if len(result['prices']) >= 15:
        closes = [p['close'] for p in result['prices']]
        rsi = calc.calculate_rsi(closes, 14)
        print(f"\n📊 技术指标：")
        print(f"  RSI(14)：{rsi:.2f}")
        
        # 检查预警条件
        print(f"\n🔍 预警条件检查：")
        print(f"  回撤 ≥ 30%？ {drawdown >= 30} (当前：{drawdown:.2f}%)")
        if drawdown >= 30:
            print(f"  RSI < 35？ {rsi < 35} (当前：{rsi:.2f})")
            print(f"  RSI < 30？ {rsi < 30} (当前：{rsi:.2f})")
            
            if drawdown >= 50 and rsi < 30:
                print(f"  ✅ 符合红色预警条件")
            elif drawdown >= 40 and rsi < 30:
                print(f"  ✅ 符合橙色预警条件")
            elif drawdown >= 30 and rsi < 35:
                print(f"  ✅ 符合黄色预警条件")
            else:
                print(f"  ❌ 未达到预警阈值")
                print(f"     原因：回撤{drawdown:.1f}%但RSI={rsi:.1f}（不够超卖）")
        else:
            print(f"  ❌ 回撤不足30%，无需进入预警流程")
    
    print(f"\n📈 最近10日收盘价：")
    for p in result['prices'][-10:]:
        print(f"  {p['date']}: HK$ {p['close']:.2f}")
        
else:
    print(f"❌ 获取失败: {result['error']}")

print("\n" + "="*60)
