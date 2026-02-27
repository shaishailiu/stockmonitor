#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预警分析模块
"""
from typing import Dict, List, Optional
from enum import Enum


class AlertLevel(Enum):
    """预警级别"""
    NONE = "无预警"
    YELLOW = "🟡 黄色预警"
    ORANGE = "🟠 橙色预警"
    RED = "🔴 红色预警"


class AlertAnalyzer:
    """预警分析器"""
    
    def __init__(self, config: Dict):
        self.thresholds = config['alert_thresholds']
    
    def analyze(self, asset_data: Dict) -> Dict:
        """
        分析单个资产，返回预警信息
        
        Args:
            asset_data: {
                'name': 资产名称,
                'current_price': 当前价,
                'high_52w': 52周高点,
                'drawdown': 回撤幅度,
                'rsi': RSI指标,
                'weekly_rsi': 周线RSI,
                'pe': 市盈率,
                'pb': 市净率,
                'pe_percentile': PE历史分位,
                'pb_percentile': PB历史分位,
                'volume_ratio': 成交量比率,
                ...
            }
            
        Returns:
            {
                'level': AlertLevel,
                'reasons': [触发原因列表],
                'score': 综合评分,
                'recommendation': 操作建议
            }
        """
        drawdown = asset_data.get('drawdown', 0)
        rsi = asset_data.get('rsi', 50)
        weekly_rsi = asset_data.get('weekly_rsi', 50)
        pe_percentile = asset_data.get('pe_percentile', 50)
        pb_percentile = asset_data.get('pb_percentile', 50)
        volume_ratio = asset_data.get('volume_ratio', 1.0)
        
        # 检查红色预警
        red_reasons = []
        if drawdown >= self.thresholds['red']['drawdown']:
            if weekly_rsi < self.thresholds['red']['weekly_rsi']:
                red_reasons.append(f"回撤{drawdown:.1f}% + 周线RSI<{self.thresholds['red']['weekly_rsi']}")
            
            if pe_percentile > 0 and pe_percentile < self.thresholds['red']['valuation_percentile']:
                red_reasons.append(f"回撤{drawdown:.1f}% + 估值分位<{self.thresholds['red']['valuation_percentile']}%")
        
        if red_reasons:
            return {
                'level': AlertLevel.RED,
                'reasons': red_reasons,
                'score': self._calculate_score(drawdown, rsi, pe_percentile),
                'recommendation': '💎 重点关注，可分批建仓（20-30%）'
            }
        
        # 检查橙色预警
        orange_reasons = []
        if drawdown >= self.thresholds['orange']['drawdown']:
            if (rsi < self.thresholds['orange']['rsi'] and 
                weekly_rsi < self.thresholds['orange']['weekly_rsi']):
                orange_reasons.append(f"回撤{drawdown:.1f}% + RSI<{self.thresholds['orange']['rsi']} + 周线RSI<{self.thresholds['orange']['weekly_rsi']}")
            
            if pe_percentile > 0 and pe_percentile < self.thresholds['orange']['valuation_percentile']:
                orange_reasons.append(f"回撤{drawdown:.1f}% + 估值分位<{self.thresholds['orange']['valuation_percentile']}%")
            
            if volume_ratio < self.thresholds['orange']['volume_ratio']:
                orange_reasons.append(f"回撤{drawdown:.1f}% + 缩量({volume_ratio:.1%})")
        
        if orange_reasons:
            return {
                'level': AlertLevel.ORANGE,
                'reasons': orange_reasons,
                'score': self._calculate_score(drawdown, rsi, pe_percentile),
                'recommendation': '👀 持续关注，可小仓位试探（10%）'
            }
        
        # 检查黄色预警
        yellow_reasons = []
        if drawdown >= self.thresholds['yellow']['drawdown']:
            if rsi < self.thresholds['yellow']['rsi']:
                yellow_reasons.append(f"回撤{drawdown:.1f}% + RSI<{self.thresholds['yellow']['rsi']}")
            
            if pe_percentile > 0 and pe_percentile < self.thresholds['yellow']['valuation_percentile']:
                yellow_reasons.append(f"回撤{drawdown:.1f}% + 估值分位<{self.thresholds['yellow']['valuation_percentile']}%")
        
        if yellow_reasons:
            return {
                'level': AlertLevel.YELLOW,
                'reasons': yellow_reasons,
                'score': self._calculate_score(drawdown, rsi, pe_percentile),
                'recommendation': '📝 进入观察，暂不操作'
            }
        
        # 无预警
        return {
            'level': AlertLevel.NONE,
            'reasons': [],
            'score': 0,
            'recommendation': ''
        }
    
    def _calculate_score(self, drawdown: float, rsi: float, valuation_percentile: float) -> int:
        """
        计算综合评分 (0-100)
        分数越高，表示机会越好
        """
        score = 0
        
        # 回撤维度 (0-40分)
        if drawdown >= 50:
            score += 40
        elif drawdown >= 40:
            score += 30
        elif drawdown >= 30:
            score += 20
        
        # 超卖维度 (0-30分)
        if rsi < 20:
            score += 30
        elif rsi < 25:
            score += 25
        elif rsi < 30:
            score += 20
        elif rsi < 35:
            score += 15
        
        # 估值维度 (0-30分)
        if valuation_percentile > 0:
            if valuation_percentile < 10:
                score += 30
            elif valuation_percentile < 15:
                score += 25
            elif valuation_percentile < 20:
                score += 20
            elif valuation_percentile < 30:
                score += 15
        
        return min(score, 100)
    
    def check_fundamental_risk(self, asset_data: Dict) -> Dict:
        """
        基本面排雷检查
        
        Returns:
            {
                'has_risk': True/False,
                'risk_factors': [风险因素列表],
                'downgrade': True/False  # 是否需要降级
            }
        """
        # 对于加密货币和贵金属，跳过基本面检查
        if asset_data.get('asset_type') in ['crypto', 'commodity']:
            return {
                'has_risk': False,
                'risk_factors': [],
                'downgrade': False
            }
        
        risk_factors = []
        
        # 检查营收下滑
        revenue_decline = asset_data.get('revenue_decline_pct', 0)
        if revenue_decline > 30:
            risk_factors.append(f"营收同比下滑{revenue_decline:.1f}%")
        
        # 检查现金流
        if asset_data.get('negative_cashflow_quarters', 0) >= 2:
            risk_factors.append("经营现金流连续2季为负")
        
        # 检查资产负债率
        debt_ratio = asset_data.get('debt_ratio', 0)
        industry_avg = asset_data.get('industry_debt_avg', 100)
        if debt_ratio > industry_avg * 1.5:
            risk_factors.append(f"负债率{debt_ratio:.1f}%（行业均值{industry_avg:.1f}%）")
        
        # 检查质押比例
        pledge_ratio = asset_data.get('pledge_ratio', 0)
        if pledge_ratio > 60:
            risk_factors.append(f"大股东质押比例{pledge_ratio:.1f}%")
        
        # 检查重大利空
        if asset_data.get('has_major_negative', False):
            risk_factors.append("存在重大利空消息")
        
        return {
            'has_risk': len(risk_factors) > 0,
            'risk_factors': risk_factors,
            'downgrade': len(risk_factors) >= 2  # 2个以上风险因素则降级
        }


class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def generate_alert_message(asset: Dict, alert: Dict, fundamental_check: Dict) -> str:
        """
        生成预警消息
        
        Args:
            asset: 资产基础信息
            alert: 预警分析结果
            fundamental_check: 基本面检查结果
        """
        if alert['level'] == AlertLevel.NONE:
            return ""
        
        # 构建富途链接
        code = asset.get('code', '')
        market = asset.get('market', '')
        if market == 'HK':
            link = f"https://www.futunn.com/stock/{code}-HK"
        elif market == 'SH':
            link = f"https://www.futunn.com/stock/{code}-SH"
        elif market == 'SZ':
            link = f"https://www.futunn.com/stock/{code}-SZ"
        elif market == 'US':
            link = f"https://www.futunn.com/stock/{code}-US"
        else:
            link = ""
        
        # 构建消息
        lines = []
        lines.append(f"{alert['level'].value} [{asset['name']}]({link}) 触发底部信号")
        lines.append("")
        
        # 触发原因
        lines.append("🎯 触发原因：")
        for reason in alert['reasons']:
            lines.append(f"  • {reason}")
        lines.append("")
        
        # 价格信息
        lines.append("📉 价格信息：")
        lines.append(f"  • 当前价：¥{asset.get('current_price', 0):.2f}")
        lines.append(f"  • 52周高点：¥{asset.get('high_52w', 0):.2f}")
        lines.append(f"  • 回撤幅度：{asset.get('drawdown', 0):.1f}%")
        lines.append("")
        
        # 估值数据（如果有）
        if asset.get('pe') or asset.get('pb'):
            lines.append("📊 估值数据：")
            if asset.get('pe'):
                lines.append(f"  • PE(TTM)：{asset['pe']:.2f} | 历史分位：{asset.get('pe_percentile', 0):.1f}%")
            if asset.get('pb'):
                lines.append(f"  • PB：{asset['pb']:.2f} | 历史分位：{asset.get('pb_percentile', 0):.1f}%")
            lines.append("")
        
        # 技术指标
        lines.append("📈 技术指标：")
        if asset.get('rsi'):
            lines.append(f"  • RSI(14)：{asset['rsi']:.1f} | 周线RSI：{asset.get('weekly_rsi', 0):.1f}")
        if asset.get('volume_ratio'):
            lines.append(f"  • 成交量：{asset['volume_ratio']:.1%}（相对60日均量）")
        lines.append("")
        
        # 基本面检查
        lines.append("⚠️  基本面检查：")
        if fundamental_check['has_risk']:
            lines.append("  ❌ 存在风险因素：")
            for risk in fundamental_check['risk_factors']:
                lines.append(f"    - {risk}")
            if fundamental_check['downgrade']:
                lines.append("  ⚠️  建议降低仓位或延后操作")
        else:
            lines.append("  ✅ 未发现明显风险")
        lines.append("")
        
        # 综合评分
        lines.append(f"💯 综合评分：{alert['score']}/100")
        lines.append("")
        
        # 操作建议
        lines.append(f"💡 {alert['recommendation']}")
        lines.append("")
        
        # 时间戳
        from datetime import datetime
        lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_summary_report(alerts: List[Dict]) -> str:
        """生成每日汇总报告"""
        from datetime import datetime
        
        if not alerts:
            return "✅ 今日无新预警触发"
        
        # 按级别分组
        red_alerts = [a for a in alerts if a['alert']['level'] == AlertLevel.RED]
        orange_alerts = [a for a in alerts if a['alert']['level'] == AlertLevel.ORANGE]
        yellow_alerts = [a for a in alerts if a['alert']['level'] == AlertLevel.YELLOW]
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"📊 投资监控日报 - {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("=" * 60)
        lines.append("")
        
        if red_alerts:
            lines.append(f"🔴 红色预警 ({len(red_alerts)}个)：")
            for a in red_alerts:
                lines.append(f"  • {a['asset']['name']} - 回撤{a['asset']['drawdown']:.1f}% - 评分{a['alert']['score']}")
            lines.append("")
        
        if orange_alerts:
            lines.append(f"🟠 橙色预警 ({len(orange_alerts)}个)：")
            for a in orange_alerts:
                lines.append(f"  • {a['asset']['name']} - 回撤{a['asset']['drawdown']:.1f}% - 评分{a['alert']['score']}")
            lines.append("")
        
        if yellow_alerts:
            lines.append(f"🟡 黄色预警 ({len(yellow_alerts)}个)：")
            for a in yellow_alerts:
                lines.append(f"  • {a['asset']['name']} - 回撤{a['asset']['drawdown']:.1f}% - 评分{a['alert']['score']}")
            lines.append("")
        
        lines.append(f"共触发 {len(alerts)} 个预警信号")
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == '__main__':
    # 测试代码
    import json
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    analyzer = AlertAnalyzer(config)
    
    # 模拟数据
    test_asset = {
        'name': '测试股票',
        'code': '000001',
        'market': 'SZ',
        'current_price': 70,
        'high_52w': 100,
        'drawdown': 30,
        'rsi': 32,
        'weekly_rsi': 28,
        'pe': 15,
        'pe_percentile': 12,
        'pb': 2.5,
        'pb_percentile': 18,
        'volume_ratio': 0.45
    }
    
    alert = analyzer.analyze(test_asset)
    print(f"预警级别: {alert['level'].value}")
    print(f"触发原因: {alert['reasons']}")
    print(f"综合评分: {alert['score']}")
