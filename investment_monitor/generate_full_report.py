#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成完整的预警报告
"""
import json
from data_fetcher_v2 import HybridDataFetcher, IndicatorCalculator
from alert_analyzer import AlertAnalyzer, ReportGenerator, AlertLevel

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

fetcher = HybridDataFetcher()
calc = IndicatorCalculator()
analyzer = AlertAnalyzer(config)
report_gen = ReportGenerator()

# 获取美团、比亚迪、小鹏的详细数据
assets_to_analyze = [
    {'symbol': 'hk03690', 'name': '美团', 'code': '03690', 'market': 'HK', 'asset_type': 'hk_stock'},
    {'symbol': 'sz002594', 'name': '比亚迪', 'code': '002594', 'market': 'SZ', 'asset_type': 'a_stock'},
    {'symbol': 'hk09868', 'name': '小鹏汽车', 'code': '09868', 'market': 'HK', 'asset_type': 'hk_stock'},
]

print("="*60)
print("生成详细预警报告")
print("="*60)

for asset in assets_to_analyze:
    print(f"\n{'='*60}")
    print(f"分析: {asset['name']} ({asset['symbol']})")
    print(f"{'='*60}")
    
    # 获取数据
    result = fetcher.fetch_stock_data(asset['symbol'], asset['market'])
    
    if not result['success']:
        print(f"❌ 获取数据失败")
        continue
    
    # 计算技术指标
    drawdown = calc.calculate_drawdown(result['current_price'], result['high_52w'])
    
    closes = [p['close'] for p in result['prices']]
    rsi = calc.calculate_rsi(closes, 14) if len(closes) >= 15 else 50
    
    # 构建完整数据
    asset_data = {
        **asset,
        'current_price': result['current_price'],
        'high_52w': result['high_52w'],
        'low_52w': result['low_52w'],
        'high_date': result['high_date'],
        'drawdown': drawdown,
        'rsi': rsi,
        'weekly_rsi': rsi * 0.95,
        'pe': 0,
        'pb': 0,
        'pe_percentile': 0,
        'pb_percentile': 0,
        'volume_ratio': 1.0
    }
    
    # 预警分析
    alert = analyzer.analyze(asset_data)
    
    print(f"预警级别: {alert['level'].value}")
    print(f"综合评分: {alert['score']}/100")
    
    if alert['level'] != AlertLevel.NONE:
        # 基本面检查
        fundamental = analyzer.check_fundamental_risk(asset_data)
        
        # 生成完整预警消息
        message = report_gen.generate_alert_message(asset_data, alert, fundamental)
        
        print("\n" + "="*60)
        print("完整预警消息:")
        print("="*60)
        print(message)
        print("="*60)
    else:
        print("✅ 未触发预警")

print("\n\n" + "="*60)
print("报告生成完成")
print("="*60)
