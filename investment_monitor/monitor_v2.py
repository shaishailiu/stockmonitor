#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资监控主程序 v2 - 混合数据源版本
"""
import json
import sys
import argparse
from datetime import datetime
from typing import List, Dict
import time

from data_fetcher_v2 import HybridDataFetcher, IndicatorCalculator
from data_fetcher import DataCache
from alert_analyzer import AlertAnalyzer, ReportGenerator, AlertLevel


class InvestmentMonitorV2:
    """投资监控主类 - 混合数据源版本"""
    
    def __init__(self, config_path: str = './config.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.fetcher = HybridDataFetcher()
        self.calculator = IndicatorCalculator()
        self.cache = DataCache('./data')
        self.analyzer = AlertAnalyzer(self.config)
        self.report_gen = ReportGenerator()
    
    def scan_all_assets(self, test_mode: bool = False) -> List[Dict]:
        """扫描所有资产"""
        print("=" * 60)
        print(f"开始扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        all_assets = []
        
        # 合并所有资产
        for asset in self.config['assets']['a_stock']:
            asset['asset_type'] = 'a_stock'
            asset['market'] = 'A'
            all_assets.append(asset)
        
        for asset in self.config['assets']['hk_stock']:
            asset['asset_type'] = 'hk_stock'
            asset['market'] = 'HK'
            all_assets.append(asset)
        
        # 添加美股
        if 'us_stock' in self.config['assets']:
            for asset in self.config['assets']['us_stock']:
                asset['asset_type'] = 'us_stock'
                asset['market'] = 'US'
                all_assets.append(asset)
        
        if test_mode:
            all_assets = all_assets[:5]
            print(f"⚠️  测试模式：仅扫描前 {len(all_assets)} 个资产")
            print()
        
        alerts = []
        
        for i, asset in enumerate(all_assets, 1):
            print(f"[{i}/{len(all_assets)}] 扫描: {asset['name']} ({asset['symbol']}-{asset['market']})...")
            
            try:
                # 1. 获取数据（自动选择数据源）
                result = self.fetcher.fetch_stock_data(asset['symbol'], asset['market'])
                
                if not result['success']:
                    print(f"  ❌ 获取数据失败: {result.get('error', '未知错误')}")
                    continue
                
                # 2. 保存到缓存（用于港股累积）
                if asset['market'] == 'HK':
                    self.cache.save_daily_data(asset['symbol'], {
                        'current_price': result['current_price'],
                        'high': result['current_price'],
                        'low': result['current_price'],
                    })
                
                # 3. 计算回撤
                drawdown = self.calculator.calculate_drawdown(
                    result['current_price'],
                    result['high_52w']
                )
                
                # 4. 构建完整资产数据
                asset_data = {
                    **asset,
                    'current_price': result['current_price'],
                    'high_52w': result['high_52w'],
                    'low_52w': result['low_52w'],
                    'high_date': result['high_date'],
                    'drawdown': drawdown,
                    'data_source': result['source'],
                }
                
                print(f"  💰 价格: ${result['current_price']:.2f} | 52W高点: ${result['high_52w']:.2f} ({result['high_date']}) | 回撤: {drawdown:.1f}%")
                print(f"  📡 数据源: {result['source']}")
                
                # 5. 如果回撤>=30%，进行深度分析
                if drawdown >= 30:
                    print(f"  ⚠️  回撤达到{drawdown:.1f}%，进入深度分析...")
                    
                    # 计算RSI
                    if len(result['prices']) >= 14:
                        close_prices = [p['close'] for p in result['prices']]
                        rsi = self.calculator.calculate_rsi(close_prices, 14)
                        asset_data['rsi'] = rsi if rsi else 50
                        asset_data['weekly_rsi'] = rsi * 0.95 if rsi else 50
                        print(f"    📊 RSI(14): {asset_data['rsi']:.1f}")
                    else:
                        asset_data['rsi'] = 50
                        asset_data['weekly_rsi'] = 50
                        print(f"    ⚠️  历史数据不足，RSI使用默认值")
                    
                    # 估值指标（占位）
                    asset_data['pe'] = 0
                    asset_data['pb'] = 0
                    asset_data['pe_percentile'] = 0
                    asset_data['pb_percentile'] = 0
                    asset_data['volume_ratio'] = 1.0
                    
                    # 6. 预警分析
                    alert = self.analyzer.analyze(asset_data)
                    
                    if alert['level'] != AlertLevel.NONE:
                        print(f"  🚨 {alert['level'].value} | 评分: {alert['score']}")
                        
                        fundamental_check = self.analyzer.check_fundamental_risk(asset_data)
                        
                        alerts.append({
                            'asset': asset_data,
                            'alert': alert,
                            'fundamental': fundamental_check
                        })
                    else:
                        print(f"  ✅ 未触发预警")
                else:
                    print(f"  ✅ 回撤较小，无需深度分析")
                
                # Alpha Vantage 限流控制
                if result.get('source') == 'AlphaVantage':
                    time.sleep(1)
                else:
                    time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                continue
        
        print()
        print("=" * 60)
        print(f"扫描完成 - 触发预警: {len(alerts)} 个")
        print("=" * 60)
        
        return alerts
    
    def send_notifications(self, alerts: List[Dict]):
        """发送预警通知（完整详细版）"""
        if not alerts:
            # 即使无预警也发送每日报告
            from datetime import datetime
            no_alert_msg = f"""📊 投资监控日报 - {datetime.now().strftime('%Y-%m-%d')}

✅ 今日扫描完成，暂无新预警触发

扫描资产：22只（5只A股 + 12只港股 + 5只美股）
数据源：腾讯财经 + Alpha Vantage

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            print(no_alert_msg)
            # 发送到企业微信
            # self._send_message(no_alert_msg)
            return
        
        level_order = {AlertLevel.RED: 0, AlertLevel.ORANGE: 1, AlertLevel.YELLOW: 2}
        alerts_sorted = sorted(alerts, key=lambda x: level_order.get(x['alert']['level'], 999))
        
        print("\n" + "="*60)
        print("开始发送完整预警报告...")
        print("="*60)
        
        # 发送每个预警的详细消息
        for i, item in enumerate(alerts_sorted, 1):
            asset = item['asset']
            alert = item['alert']
            fundamental = item['fundamental']
            
            message = self.report_gen.generate_alert_message(asset, alert, fundamental)
            
            print(f"\n{'='*60}")
            print(f"[{i}/{len(alerts)}] {alert['level'].value} - {asset['name']}")
            print("="*60)
            print(message)
            print("="*60)
            
            # 实际发送到企业微信
            self._send_message(message)
            
            time.sleep(2)  # 避免刷屏
        
        # 发送汇总报告
        from datetime import datetime
        summary_header = f"""{'='*60}
📊 投资监控完整扫描报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

扫描资产：22只（5只A股 + 12只港股 + 5只美股）
数据源：腾讯财经（A股+港股完整52周）+ Alpha Vantage（美股100天）
"""
        
        summary = self.report_gen.generate_summary_report(alerts_sorted)
        full_summary = summary_header + "\n" + summary
        
        print("\n" + full_summary)
        self._send_message(full_summary)
    
    def _send_message(self, message: str):
        """通过 OpenClaw message 工具发送消息"""
        import subprocess
        
        channel = self.config['notification']['channel']
        target = self.config['notification']['target']
        
        # 使用 message 工具发送
        cmd = f"openclaw message send --channel {channel} --target {target} --message '{message}'"
        
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"  ✅ 消息发送成功")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 消息发送失败: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(description='长线投资监控系统 v2 - 混合数据源')
    parser.add_argument('--test', action='store_true', help='测试模式（仅扫描前5个资产）')
    parser.add_argument('--no-notify', action='store_true', help='不发送通知，仅打印结果')
    
    args = parser.parse_args()
    
    try:
        monitor = InvestmentMonitorV2()
        alerts = monitor.scan_all_assets(test_mode=args.test)
        
        if not args.no_notify:
            monitor.send_notifications(alerts)
        
        print("\n✅ 监控完成")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
