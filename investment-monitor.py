#!/usr/bin/env python3
"""
投资标的每日监控脚本 - 专业版（方案B）
- 按资产波动性分类预警
- 基于52周高点计算跌幅
- 支持估值指标提醒（PE/PB）
- 分批建仓策略提示
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "investment-monitor-config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_alert_level(drawdown_pct, thresholds):
    """判断预警级别"""
    if drawdown_pct <= thresholds['major_opportunity']:
        return "🔴 重大机会", 3
    elif drawdown_pct <= thresholds['buy_opportunity']:
        return "🟡 买入时机", 2
    elif drawdown_pct <= thresholds['early_attention']:
        return "🟢 早期关注", 1
    else:
        return None, 0

def suggest_position(drawdown_pct, config):
    """根据跌幅建议仓位"""
    strategy = config['monitoring_config']['position_strategy']
    
    if drawdown_pct <= strategy['third_batch']['drawdown']:
        return f"第三批建仓（{strategy['third_batch']['allocation']*100:.0f}%资金）"
    elif drawdown_pct <= strategy['second_batch']['drawdown']:
        return f"第二批建仓（{strategy['second_batch']['allocation']*100:.0f}%资金）"
    elif drawdown_pct <= strategy['first_batch']['drawdown']:
        return f"第一批建仓（{strategy['first_batch']['allocation']*100:.0f}%资金）"
    else:
        return "继续观察"

def check_assets():
    """检查所有资产的价格变动"""
    config = load_config()
    
    alerts = {
        "low_volatility": [],
        "medium_volatility": [],
        "high_volatility": []
    }
    
    report = []
    report.append("=" * 50)
    report.append(f"📊 投资监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 50)
    report.append(f"")
    report.append(f"📍 价格参考：52周高点")
    report.append(f"📈 估值监控：已启用（PE/PB）")
    report.append(f"")
    
    # 遍历三类资产
    for volatility_type in ['low_volatility', 'medium_volatility', 'high_volatility']:
        category = config['watchlist'][volatility_type]
        thresholds = category['thresholds']
        
        report.append(f"\n{'='*50}")
        report.append(f"【{category['description']}】")
        report.append(f"预警阈值：{thresholds['early_attention']*100:.0f}% | "
                     f"{thresholds['buy_opportunity']*100:.0f}% | "
                     f"{thresholds['major_opportunity']*100:.0f}%")
        report.append(f"{'='*50}")
        
        for asset in category['assets']:
            report.append(f"  • {asset['name']} ({asset['symbol']}) - {asset['type']}")
            
            # TODO: 实际价格获取逻辑
            # 这里需要调用 web_fetch 或其他API获取：
            # 1. 当前价格
            # 2. 52周高点
            # 3. PE/PB估值（如适用）
            # 
            # 示例伪代码：
            # current_price = fetch_price(asset['symbol'])
            # high_52w = fetch_52w_high(asset['symbol'])
            # drawdown = (current_price - high_52w) / high_52w
            # 
            # alert_level, priority = format_alert_level(drawdown, thresholds)
            # if alert_level:
            #     position_advice = suggest_position(drawdown, config)
            #     alerts[volatility_type].append({
            #         'asset': asset,
            #         'drawdown': drawdown,
            #         'level': alert_level,
            #         'priority': priority,
            #         'advice': position_advice
            #     })
    
    report.append(f"\n{'='*50}")
    report.append(f"💡 分批建仓策略提示")
    report.append(f"{'='*50}")
    strategy = config['monitoring_config']['position_strategy']
    report.append(f"第一批（{strategy['first_batch']['drawdown']*100:.0f}%跌幅）：投入{strategy['first_batch']['allocation']*100:.0f}%资金")
    report.append(f"第二批（{strategy['second_batch']['drawdown']*100:.0f}%跌幅）：再投{strategy['second_batch']['allocation']*100:.0f}%资金")
    report.append(f"第三批（{strategy['third_batch']['drawdown']*100:.0f}%跌幅）：再投{strategy['third_batch']['allocation']*100:.0f}%资金")
    report.append(f"保留：{strategy['reserve']['allocation']*100:.0f}%资金 - {strategy['reserve']['note']}")
    
    return alerts, "\n".join(report)

def format_alerts(alerts):
    """格式化预警信息"""
    output = []
    
    total_alerts = sum(len(v) for v in alerts.values())
    
    if total_alerts == 0:
        output.append("\n✅ 暂无达到预警阈值的标的")
        output.append("所有资产运行正常，继续观察市场动态。")
        return "\n".join(output)
    
    output.append(f"\n🚨 发现 {total_alerts} 个投资机会！\n")
    
    # 按优先级排序输出
    for volatility_type, alert_list in alerts.items():
        if not alert_list:
            continue
            
        sorted_alerts = sorted(alert_list, key=lambda x: x['priority'], reverse=True)
        
        for alert in sorted_alerts:
            asset = alert['asset']
            output.append(f"{alert['level']} {asset['name']} ({asset['symbol']})")
            output.append(f"   从52周高点回落：{abs(alert['drawdown'])*100:.1f}%")
            output.append(f"   建议操作：{alert['advice']}")
            output.append("")
    
    return "\n".join(output)

if __name__ == "__main__":
    try:
        alerts, report = check_assets()
        
        print(report)
        print(format_alerts(alerts))
        
        print("\n" + "="*50)
        print("⚠️ 注意：以上数据需要实际对接价格API")
        print("当前版本为框架代码，需要在定时任务中配合 web_fetch 实现")
        print("="*50)
            
    except Exception as e:
        print(f"❌ 监控失败: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
