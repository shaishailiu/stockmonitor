#!/usr/bin/env python3
"""
长线投资底部监控系统
使用 AkShare 获取实时行情和估值数据
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# 配置文件路径
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(CONFIG_DIR, "monitor_data.json")
ALERT_HISTORY = os.path.join(CONFIG_DIR, "alert_history.json")

# 监控资产配置
ASSETS = {
    "crypto": [
        {"symbol": "BTC", "name": "比特币", "type": "crypto"}
    ],
    "metals": [
        {"symbol": "XAU", "name": "黄金", "type": "metal"}
    ],
    "us_tech": [
        {"symbol": "AAPL", "name": "苹果"},
        {"symbol": "MSFT", "name": "微软"},
        {"symbol": "GOOGL", "name": "谷歌"},
        {"symbol": "AMZN", "name": "亚马逊"},
        {"symbol": "NVDA", "name": "英伟达"},
        {"symbol": "META", "name": "Meta"},
        {"symbol": "TSLA", "name": "特斯拉"}
    ],
    "semiconductor": [
        {"symbol": "AMD", "name": "AMD"},
        {"symbol": "TSM", "name": "台积电"},
        {"symbol": "AVGO", "name": "博通"},
        {"symbol": "QCOM", "name": "高通"},
        {"symbol": "ASML", "name": "阿斯麦"},
        {"symbol": "INTC", "name": "英特尔"},
        {"symbol": "0981.HK", "name": "中芯国际", "market": "HK"},
        {"symbol": "1347.HK", "name": "华虹半导体", "market": "HK"}
    ],
    "china_us": [
        {"symbol": "0700.HK", "name": "腾讯", "market": "HK"},
        {"symbol": "BABA", "name": "阿里巴巴"},
        {"symbol": "JD", "name": "京东"},
        {"symbol": "PDD", "name": "拼多多"},
        {"symbol": "3690.HK", "name": "美团", "market": "HK"},
        {"symbol": "NTES", "name": "网易"},
        {"symbol": "BIDU", "name": "百度"},
        {"symbol": "TCOM", "name": "携程"},
        {"symbol": "BILI", "name": "B站"},
        {"symbol": "TME", "name": "腾讯音乐"},
        {"symbol": "XPEV", "name": "小鹏汽车"},
        {"symbol": "NIO", "name": "蔚来"},
        {"symbol": "LI", "name": "理想汽车"},
        {"symbol": "FUTU", "name": "富途"},
        {"symbol": "TIGR", "name": "老虎证券"},
        {"symbol": "BEKE", "name": "贝壳"},
        {"symbol": "MNSO", "name": "名创优品"},
        {"symbol": "EDU", "name": "新东方"}
    ],
    "hk_tech": [
        {"symbol": "1810.HK", "name": "小米", "market": "HK"},
        {"symbol": "1024.HK", "name": "快手", "market": "HK"},
        {"symbol": "0772.HK", "name": "阅文集团", "market": "HK"},
        {"symbol": "1211.HK", "name": "比亚迪", "market": "HK"}
    ],
    "a_share": [
        {"symbol": "600519", "name": "贵州茅台", "market": "SH"},
        {"symbol": "300750", "name": "宁德时代", "market": "SZ"},
        {"symbol": "002594", "name": "比亚迪", "market": "SZ"},
        {"symbol": "600036", "name": "招商银行", "market": "SH"},
        {"symbol": "601318", "name": "中国平安", "market": "SH"}
    ]
}


class InvestmentMonitor:
    """投资监控主类"""
    
    def __init__(self):
        self.alerts = []
        self.load_history()
    
    def load_history(self):
        """加载历史预警数据"""
        if os.path.exists(ALERT_HISTORY):
            with open(ALERT_HISTORY, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        else:
            self.history = {}
    
    def save_history(self):
        """保存预警历史"""
        with open(ALERT_HISTORY, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def get_us_stock_data(self, symbol):
        """获取美股数据"""
        try:
            # 获取实时行情
            df = ak.stock_us_spot_em()
            stock = df[df['代码'] == symbol]
            
            if stock.empty:
                return None
            
            current_price = float(stock['最新价'].values[0])
            
            # 获取历史数据计算52周高点
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            hist_df = ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date)
            
            if hist_df.empty:
                return None
            
            high_52w = hist_df['高'].max()
            drawdown = ((high_52w - current_price) / high_52w) * 100
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "high_52w": high_52w,
                "drawdown": drawdown,
                "volume": float(stock['成交量'].values[0]) if '成交量' in stock.columns else 0
            }
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def get_hk_stock_data(self, symbol):
        """获取港股数据"""
        try:
            # 港股代码格式转换
            code = symbol.replace('.HK', '')
            
            # 获取实时行情
            df = ak.stock_hk_spot_em()
            stock = df[df['代码'] == code]
            
            if stock.empty:
                return None
            
            current_price = float(stock['最新价'].values[0])
            
            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            hist_df = ak.stock_hk_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
            
            if hist_df.empty:
                return None
            
            high_52w = hist_df['高'].max()
            drawdown = ((high_52w - current_price) / high_52w) * 100
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "high_52w": high_52w,
                "drawdown": drawdown
            }
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def get_a_share_data(self, symbol, market):
        """获取A股数据"""
        try:
            # A股实时行情
            df = ak.stock_zh_a_spot_em()
            stock = df[df['代码'] == symbol]
            
            if stock.empty:
                return None
            
            current_price = float(stock['最新价'].values[0])
            
            # 历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            hist_df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
            
            if hist_df.empty:
                return None
            
            high_52w = hist_df['高'].max()
            drawdown = ((high_52w - current_price) / high_52w) * 100
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "high_52w": high_52w,
                "drawdown": drawdown,
                "market": market
            }
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def get_crypto_data(self, symbol):
        """获取加密货币数据"""
        try:
            # 获取比特币行情
            df = ak.crypto_hist(symbol="btcusdt")
            
            if df.empty:
                return None
            
            current_price = float(df.iloc[-1]['收盘'])
            high_52w = df['最高'].tail(365).max()
            drawdown = ((high_52w - current_price) / high_52w) * 100
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "high_52w": high_52w,
                "drawdown": drawdown,
                "type": "crypto"
            }
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def get_gold_data(self):
        """获取黄金数据"""
        try:
            df = ak.futures_global_em(symbol="黄金连续")
            
            if df.empty:
                return None
            
            current_price = float(df.iloc[-1]['收盘价'])
            high_52w = df['最高价'].tail(365).max()
            drawdown = ((high_52w - current_price) / high_52w) * 100
            
            return {
                "symbol": "XAU",
                "current_price": current_price,
                "high_52w": high_52w,
                "drawdown": drawdown,
                "type": "metal"
            }
        except Exception as e:
            print(f"获取黄金数据失败: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_deep_analysis(self, symbol, market=None):
        """深度分析（回撤>=30%时触发）"""
        try:
            analysis = {}
            
            # 获取历史数据用于技术指标计算
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=200)).strftime('%Y%m%d')
            
            if market == "HK":
                code = symbol.replace('.HK', '')
                hist_df = ak.stock_hk_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
            elif market in ["SH", "SZ"]:
                hist_df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
            else:
                # 美股
                hist_df = ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date)
            
            if not hist_df.empty:
                prices = hist_df['收盘'].values
                analysis['rsi_14'] = self.calculate_rsi(prices, 14)
                
                # 周线RSI (简化计算，取最近5天)
                if len(prices) >= 70:
                    weekly_prices = prices[::5]
                    analysis['rsi_weekly'] = self.calculate_rsi(weekly_prices, 14)
                
                # 成交量对比
                volumes = hist_df['成交量'].values
                if len(volumes) >= 60:
                    current_vol = volumes[-1]
                    avg_vol_60 = np.mean(volumes[-60:])
                    analysis['volume_ratio'] = (current_vol / avg_vol_60) * 100
            
            return analysis
        except Exception as e:
            print(f"深度分析失败 {symbol}: {e}")
            return {}
    
    def check_alert_level(self, data, analysis):
        """判断预警级别"""
        drawdown = data['drawdown']
        rsi = analysis.get('rsi_14', 100)
        rsi_weekly = analysis.get('rsi_weekly', 100)
        volume_ratio = analysis.get('volume_ratio', 100)
        
        # 红色预警
        if drawdown >= 50 and rsi_weekly < 30:
            return "🔴 红色"
        if drawdown >= 50:
            return "🔴 红色"
        
        # 橙色预警
        if drawdown >= 40 and rsi < 30 and rsi_weekly < 35:
            return "🟠 橙色"
        if drawdown >= 40 and volume_ratio < 50:
            return "🟠 橙色"
        
        # 黄色预警
        if drawdown >= 30 and rsi < 35:
            return "🟡 黄色"
        if drawdown >= 30:
            return "🟡 黄色"
        
        return None
    
    def generate_alert_message(self, asset, data, analysis, alert_level):
        """生成预警消息"""
        symbol = data['symbol']
        name = asset['name']
        
        # 生成富途链接
        if 'market' in asset and asset['market'] == "HK":
            code = symbol.replace('.HK', '')
            link = f"https://www.futunn.com/stock/{code}-HK"
        elif 'market' in asset and asset['market'] in ["SH", "SZ"]:
            link = f"https://www.futunn.com/stock/{symbol}-{asset['market']}"
        else:
            link = f"https://www.futunn.com/stock/{symbol}-US"
        
        msg = f"""【{alert_level}】[{name}]({link}) 触发底部信号

📉 价格信息：
- 当前价：{data['current_price']:.2f}
- 52周高点：{data['high_52w']:.2f}
- 回撤幅度：{data['drawdown']:.1f}%

📈 技术指标：
- RSI(14)：{analysis.get('rsi_14', 'N/A')}
- 周线RSI：{analysis.get('rsi_weekly', 'N/A')}
- 成交量：{analysis.get('volume_ratio', 'N/A'):.1f}%（相对60日均量）

💡 建议操作：
"""
        
        if "黄色" in alert_level:
            msg += "- 观察，暂不操作"
        elif "橙色" in alert_level:
            msg += "- 可小仓位试探（10%）"
        elif "红色" in alert_level:
            msg += "- 可分批建仓（20-30%）"
        
        msg += f"\n\n⏰ 监控时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return msg
    
    def scan_all_assets(self):
        """扫描所有资产"""
        print(f"\n{'='*60}")
        print(f"开始扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        all_assets = []
        for category, assets in ASSETS.items():
            all_assets.extend([(asset, category) for asset in assets])
        
        results = []
        
        for asset, category in all_assets:
            symbol = asset['symbol']
            name = asset['name']
            print(f"正在扫描: {name} ({symbol})...")
            
            # 获取基础数据
            data = None
            if asset.get('type') == 'crypto':
                data = self.get_crypto_data(symbol)
            elif asset.get('type') == 'metal':
                data = self.get_gold_data()
            elif asset.get('market') == 'HK':
                data = self.get_hk_stock_data(symbol)
            elif asset.get('market') in ['SH', 'SZ']:
                data = self.get_a_share_data(symbol, asset['market'])
            else:
                # 美股
                data = self.get_us_stock_data(symbol)
            
            if not data:
                print(f"  ❌ 获取数据失败\n")
                continue
            
            drawdown = data['drawdown']
            print(f"  当前价: {data['current_price']:.2f}, 回撤: {drawdown:.1f}%")
            
            # 如果回撤>=30%，进行深度分析
            if drawdown >= 30:
                print(f"  ⚠️  回撤>=30%，执行深度分析...")
                analysis = self.get_deep_analysis(symbol, asset.get('market'))
                
                # 判断预警级别
                alert_level = self.check_alert_level(data, analysis)
                
                if alert_level:
                    print(f"  🚨 触发 {alert_level} 预警！")
                    msg = self.generate_alert_message(asset, data, analysis, alert_level)
                    self.alerts.append({
                        "asset": name,
                        "symbol": symbol,
                        "level": alert_level,
                        "message": msg,
                        "timestamp": datetime.now().isoformat()
                    })
                    results.append((name, alert_level, drawdown))
            
            print()
        
        return results
    
    def send_alerts(self):
        """发送预警消息"""
        if not self.alerts:
            print("✅ 无新预警")
            return
        
        print(f"\n📢 共有 {len(self.alerts)} 个预警待发送\n")
        
        for alert in self.alerts:
            print(alert['message'])
            print("\n" + "-"*60 + "\n")
            
            # 保存到历史
            key = f"{alert['symbol']}_{alert['timestamp']}"
            self.history[key] = alert
        
        self.save_history()
    
    def run_daily_scan(self):
        """执行每日扫描"""
        self.alerts = []
        results = self.scan_all_assets()
        self.send_alerts()
        
        # 生成简报
        print(f"\n{'='*60}")
        print(f"扫描完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"触发预警: {len(self.alerts)} 个")
        
        if results:
            print("\n预警列表:")
            for name, level, drawdown in results:
                print(f"  {level} {name} (回撤 {drawdown:.1f}%)")
        
        print()


def main():
    """主函数"""
    monitor = InvestmentMonitor()
    monitor.run_daily_scan()


if __name__ == "__main__":
    main()
