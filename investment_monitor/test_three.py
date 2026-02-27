#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试指定的三只港股：腾讯、美团、百度
"""
from data_fetcher import TencentFinanceAPI, DataCache, IndicatorCalculator
from alert_analyzer import AlertAnalyzer, ReportGenerator
import json

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

api = TencentFinanceAPI()
cache = DataCache('./data')
calculator = IndicatorCalculator()
analyzer = AlertAnalyzer(config)
report_gen = ReportGenerator()

# 测试标的
test_assets = [
    {'symbol': 'hk00700', 'name': '腾讯控股', 'code': '00700', 'market': 'HK'},
    {'symbol': 'hk03690', 'name': '美团', 'code': '03690', 'market': 'HK'},
    {'symbol': 'hk09888', 'name': '百度集团', 'code': '09888', 'market': 'HK'},
]

print("=" * 60)
print("测试：腾讯、美团、百度 实时数据")
print("=" * 60)
print()

alerts = []

for asset in test_assets:
    print(f"{'='*60}")
    print(f"📊 {asset['name']} ({asset['symbol']})")
    print(f"{'='*60}")
    
    # 获取实时数据
    realtime = api.fetch_realtime(asset['symbol'])
    
    if not realtime['success']:
        print(f"❌ 获取失败: {realtime.get('error')}")
        continue
    
    # 保存到缓存
    cache.save_daily_data(asset['symbol'], realtime)
    
    # 获取52周高低点
    high_low = cache.get_52w_high_low(asset['symbol'])
    
    if high_low['high'] == 0:
        high_low['high'] = realtime['current_price']
        high_low['low'] = realtime['current_price']
    
    # 计算回撤
    drawdown = calculator.calculate_drawdown(
        realtime['current_price'],
        high_low['high']
    )
    
    # 构建完整数据
    asset_data = {
        **asset,
        'current_price': realtime['current_price'],
        'yesterday_close': realtime.get('yesterday_close'),
        'open': realtime.get('open'),
        'high': realtime.get('high'),
        'low': realtime.get('low'),
        'volume': realtime.get('volume'),
        'high_52w': high_low['high'],
        'low_52w': high_low['low'],
        'drawdown': drawdown,
        'asset_type': 'hk_stock'
    }
    
    # 打印基础信息
    print(f"\n💰 价格信息：")
    print(f"  当前价：HK$ {realtime['current_price']:.2f}")
    print(f"  昨收价：HK$ {realtime.get('yesterday_close', 0):.2f}")
    change = realtime['current_price'] - realtime.get('yesterday_close', realtime['current_price'])
    change_pct = (change / realtime.get('yesterday_close', realtime['current_price'])) * 100 if realtime.get('yesterday_close') else 0
    print(f"  涨跌：{change:+.2f} ({change_pct:+.2f}%)")
    print(f"  今日最高：HK$ {realtime.get('high', 0):.2f}")
    print(f"  今日最低：HK$ {realtime.get('low', 0):.2f}")
    print(f"  成交量：{realtime.get('volume', 0):,.0f}")
    
    print(f"\n📊 历史数据：")
    print(f"  52周最高：HK$ {high_low['high']:.2f}")
    print(f"  52周最低：HK$ {high_low['low']:.2f}")
    print(f"  回撤幅度：{drawdown:.2f}%")
    
    # 获取历史数据计算RSI
    historical = cache.load_historical_cache(asset['symbol'], 30)
    if len(historical) >= 14:
        prices = [d['current_price'] for d in historical]
        rsi = calculator.calculate_rsi(prices, 14)
        asset_data['rsi'] = rsi if rsi else 50
        asset_data['weekly_rsi'] = rsi * 0.95 if rsi else 50
        print(f"\n📈 技术指标：")
        print(f"  RSI(14)：{asset_data['rsi']:.2f}")
        print(f"  周线RSI（估算）：{asset_data['weekly_rsi']:.2f}")
    else:
        asset_data['rsi'] = 50
        asset_data['weekly_rsi'] = 50
        print(f"\n📈 技术指标：")
        print(f"  ⚠️  历史数据不足（{len(historical)}天），RSI使用默认值")
    
    # 估值占位
    asset_data['pe'] = 0
    asset_data['pb'] = 0
    asset_data['pe_percentile'] = 0
    asset_data['pb_percentile'] = 0
    asset_data['volume_ratio'] = 1.0
    
    # 预警分析
    alert = analyzer.analyze(asset_data)
    
    print(f"\n🚨 预警状态：")
    print(f"  级别：{alert['level'].value}")
    print(f"  评分：{alert['score']}/100")
    
    if alert['reasons']:
        print(f"  触发原因：")
        for reason in alert['reasons']:
            print(f"    • {reason}")
        print(f"  建议：{alert['recommendation']}")
    
    print()

print("=" * 60)
print("测试完成")
print("=" * 60)
