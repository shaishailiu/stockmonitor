#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资监控主程序
"""
import json
import sys
import argparse
from datetime import datetime
from typing import List, Dict
import time

from data_fetcher import TencentFinanceAPI, IndicatorCalculator, DataCache
from alert_analyzer import AlertAnalyzer, ReportGenerator, AlertLevel


class InvestmentMonitor:
    """投资监控主类"""
    
    def __init__(self, config_path: str = './config.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.api = TencentFinanceAPI()
        self.calculator = IndicatorCalculator()
        self.cache = DataCache('./data')
        self.analyzer = AlertAnalyzer(self.config)
        self.report_gen = ReportGenerator()
    
    def scan_all_assets(self, test_mode: bool = False) -> List[Dict]:
        """
        扫描所有资产
        
        Args:
            test_mode: 测试模式，只扫描少量资产
            
        Returns:
            触发预警的资产列表
        """
        print("=" * 60)
        print(f"开始扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        all_assets = []
        
        # 合并 A股 和 港股
        for asset in self.config['assets']['a_stock']:
            asset['asset_type'] = 'a_stock'
            all_assets.append(asset)
        
        for asset in self.config['assets']['hk_stock']:
            asset['asset_type'] = 'hk_stock'
            all_assets.append(asset)
        
        if test_mode:
            all_assets = all_assets[:3]
            print(f"⚠️  测试模式：仅扫描前 {len(all_assets)} 个资产")
            print()
        
        alerts = []
        
        for i, asset in enumerate(all_assets, 1):
            print(f"[{i}/{len(all_assets)}] 扫描: {asset['name']} ({asset['symbol']})...")
            
            try:
                # 1. 获取实时数据
                realtime = self.api.fetch_realtime(asset['symbol'])
                
                if not realtime['success']:
                    print(f"  ❌ 获取数据失败: {realtime.get('error', '未知错误')}")
                    continue
                
                # 2. 保存到缓存
                self.cache.save_daily_data(asset['symbol'], realtime)
                
                # 3. 获取52周高低点
                high_low = self.cache.get_52w_high_low(asset['symbol'])
                
                # 如果缓存中没有历史数据，使用今日价格作为基准
                if high_low['high'] == 0:
                    high_low['high'] = realtime['current_price']
                    high_low['low'] = realtime['current_price']
                    print(f"  ℹ️  首次扫描，使用当前价作为基准")
                
                # 4. 计算回撤
                drawdown = self.calculator.calculate_drawdown(
                    realtime['current_price'],
                    high_low['high']
                )
                
                # 5. 构建完整资产数据
                asset_data = {
                    **asset,
                    'current_price': realtime['current_price'],
                    'high_52w': high_low['high'],
                    'low_52w': high_low['low'],
                    'drawdown': drawdown,
                    'volume': realtime.get('volume', 0),
                }
                
                print(f"  💰 价格: ¥{realtime['current_price']:.2f} | 52W高点: ¥{high_low['high']:.2f} | 回撤: {drawdown:.1f}%")
                
                # 6. 如果回撤>=30%，进行深度分析
                if drawdown >= 30:
                    print(f"  ⚠️  回撤达到{drawdown:.1f}%，进入深度分析...")
                    
                    # 获取历史数据计算技术指标
                    historical = self.cache.load_historical_cache(asset['symbol'], 30)
                    
                    if len(historical) >= 14:
                        prices = [d['current_price'] for d in historical]
                        rsi = self.calculator.calculate_rsi(prices, 14)
                        asset_data['rsi'] = rsi if rsi else 50
                        print(f"    📊 RSI(14): {asset_data['rsi']:.1f}")
                    else:
                        asset_data['rsi'] = 50  # 默认值
                        print(f"    ⚠️  历史数据不足，RSI使用默认值")
                    
                    # 周线RSI（简化：使用日线数据估算）
                    asset_data['weekly_rsi'] = asset_data['rsi'] * 0.95  # 简化估算
                    
                    # 估值指标（暂时使用占位值，待后续集成）
                    asset_data['pe'] = 0
                    asset_data['pb'] = 0
                    asset_data['pe_percentile'] = 0
                    asset_data['pb_percentile'] = 0
                    asset_data['volume_ratio'] = 1.0
                    
                    # 7. 预警分析
                    alert = self.analyzer.analyze(asset_data)
                    
                    if alert['level'] != AlertLevel.NONE:
                        print(f"  🚨 {alert['level'].value} | 评分: {alert['score']}")
                        
                        # 8. 基本面检查
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
                
                # 延迟避免请求过快
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
        """发送预警通知"""
        if not alerts:
            print("✅ 无预警，不发送通知")
            return
        
        # 按预警级别排序（红色 > 橙色 > 黄色）
        level_order = {AlertLevel.RED: 0, AlertLevel.ORANGE: 1, AlertLevel.YELLOW: 2}
        alerts_sorted = sorted(alerts, key=lambda x: level_order.get(x['alert']['level'], 999))
        
        print("\n开始发送预警通知...")
        
        for i, item in enumerate(alerts_sorted, 1):
            asset = item['asset']
            alert = item['alert']
            fundamental = item['fundamental']
            
            # 生成消息
            message = self.report_gen.generate_alert_message(asset, alert, fundamental)
            
            print(f"\n[{i}/{len(alerts)}] 发送预警: {asset['name']}")
            print("-" * 60)
            print(message)
            print("-" * 60)
            
            # 实际发送（通过 OpenClaw message 工具）
            # 这里先打印，实际使用时取消注释
            # self._send_message(message)
            
            time.sleep(1)  # 避免刷屏
        
        # 发送汇总报告
        summary = self.report_gen.generate_summary_report(alerts_sorted)
        print("\n" + summary)
        # self._send_message(summary)
    
    def _send_message(self, message: str):
        """通过 OpenClaw 发送消息"""
        import subprocess
        
        channel = self.config['notification']['channel']
        target = self.config['notification']['target']
        
        # 调用 openclaw 的 message 命令
        cmd = [
            'openclaw', 'message', 'send',
            '--channel', channel,
            '--target', target,
            '--message', message
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print("  ✅ 消息发送成功")
        except Exception as e:
            print(f"  ❌ 消息发送失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='长线投资监控系统')
    parser.add_argument('--test', action='store_true', help='测试模式（仅扫描少量资产）')
    parser.add_argument('--no-notify', action='store_true', help='不发送通知，仅打印结果')
    
    args = parser.parse_args()
    
    try:
        monitor = InvestmentMonitor()
        
        # 扫描资产
        alerts = monitor.scan_all_assets(test_mode=args.test)
        
        # 发送通知
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
