#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示模式：模拟触发预警的场景
"""
import json
from alert_analyzer import AlertAnalyzer, ReportGenerator, AlertLevel

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

analyzer = AlertAnalyzer(config)
report_gen = ReportGenerator()

print("=" * 60)
print("投资监控系统 - 预警演示")
print("=" * 60)
print()

# 模拟几个不同级别的预警场景
test_cases = [
    {
        'name': '示例科技股A',
        'code': '00001',
        'market': 'HK',
        'current_price': 50,
        'high_52w': 100,
        'drawdown': 50,
        'rsi': 28,
        'weekly_rsi': 25,
        'pe': 15,
        'pe_percentile': 8,
        'pb': 2.5,
        'pb_percentile': 12,
        'volume_ratio': 0.35,
        'asset_type': 'hk_stock'
    },
    {
        'name': '示例消费股B',
        'code': '00002',
        'market': 'HK',
        'current_price': 60,
        'high_52w': 100,
        'drawdown': 40,
        'rsi': 32,
        'weekly_rsi': 38,
        'pe': 25,
        'pe_percentile': 18,
        'pb': 3.5,
        'pb_percentile': 22,
        'volume_ratio': 0.55,
        'asset_type': 'hk_stock'
    },
    {
        'name': '示例金融股C',
        'code': '600001',
        'market': 'SH',
        'current_price': 72,
        'high_52w': 100,
        'drawdown': 28,
        'rsi': 42,
        'weekly_rsi': 45,
        'pe': 8,
        'pe_percentile': 45,
        'pb': 1.2,
        'pb_percentile': 35,
        'volume_ratio': 0.85,
        'asset_type': 'a_stock'
    }
]

alerts = []

for asset in test_cases:
    print(f"\n{'=' * 60}")
    print(f"分析: {asset['name']}")
    print(f"{'=' * 60}")
    
    # 预警分析
    alert = analyzer.analyze(asset)
    
    print(f"预警级别: {alert['level'].value}")
    print(f"综合评分: {alert['score']}/100")
    
    if alert['level'] != AlertLevel.NONE:
        print(f"触发原因:")
        for reason in alert['reasons']:
            print(f"  • {reason}")
        
        # 基本面检查
        fundamental_check = analyzer.check_fundamental_risk(asset)
        
        # 生成完整预警消息
        message = report_gen.generate_alert_message(asset, alert, fundamental_check)
        
        print()
        print("完整预警消息:")
        print("-" * 60)
        print(message)
        print("-" * 60)
        
        alerts.append({
            'asset': asset,
            'alert': alert,
            'fundamental': fundamental_check
        })
    else:
        print("✅ 未触发预警")

# 生成汇总报告
print("\n\n" + "=" * 60)
print("汇总报告")
print("=" * 60)
summary = report_gen.generate_summary_report(alerts)
print(summary)

print("\n\n💡 提示：这是演示模式，使用模拟数据")
print("   实际监控请运行: python3 monitor.py")
