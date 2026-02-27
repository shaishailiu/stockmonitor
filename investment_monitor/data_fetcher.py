#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取模块 - 腾讯财经接口
"""
import urllib.request
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import time


class TencentFinanceAPI:
    """腾讯财经数据接口"""
    
    BASE_URL = "http://qt.gtimg.cn/q="
    
    @staticmethod
    def fetch_realtime(symbol: str) -> Optional[Dict]:
        """
        获取实时行情数据
        
        Args:
            symbol: 股票代码，如 sh600519, hk00700
            
        Returns:
            {
                'name': '股票名称',
                'code': '代码',
                'current_price': 当前价,
                'yesterday_close': 昨收价,
                'open': 开盘价,
                'high': 最高价,
                'low': 最低价,
                'volume': 成交量,
                'success': True/False
            }
        """
        try:
            url = f"{TencentFinanceAPI.BASE_URL}{symbol}"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = response.read().decode('gbk')
                
            # 解析数据
            # 格式: v_CODE="51~name~code~price~yesterday~open~volume~..."
            match = re.search(r'"(.+)"', data)
            if not match:
                return {'success': False, 'error': '数据格式错误'}
            
            fields = match.group(1).split('~')
            
            if len(fields) < 10:
                return {'success': False, 'error': '字段不完整'}
            
            # 检查是否有有效价格
            if not fields[3] or fields[3] == '0':
                return {'success': False, 'error': '无有效价格数据'}
            
            return {
                'name': fields[1],
                'code': fields[2],
                'current_price': float(fields[3]),
                'yesterday_close': float(fields[4]) if fields[4] else None,
                'open': float(fields[5]) if fields[5] else None,
                'high': float(fields[33]) if len(fields) > 33 and fields[33] else float(fields[3]),
                'low': float(fields[34]) if len(fields) > 34 and fields[34] else float(fields[3]),
                'volume': float(fields[6]) if fields[6] else 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def fetch_historical(symbol: str, days: int = 365) -> Optional[List[Dict]]:
        """
        获取历史数据（用于计算52周高点等指标）
        
        由于腾讯财经接口限制，这里使用简化方案：
        通过多次实时查询 + 缓存来模拟历史数据
        
        实际部署时建议：
        1. 使用 akshare 的历史数据接口（需要稳定网络）
        2. 或本地数据库累积历史数据
        """
        # 这里返回 None，表示需要其他方式获取历史数据
        # 实际使用时会从缓存或数据库读取
        return None
    
    @staticmethod
    def calculate_52w_high(historical_data: List[Dict]) -> float:
        """计算52周最高价"""
        if not historical_data:
            return 0.0
        
        prices = [d['high'] for d in historical_data if 'high' in d]
        return max(prices) if prices else 0.0
    
    @staticmethod
    def calculate_52w_low(historical_data: List[Dict]) -> float:
        """计算52周最低价"""
        if not historical_data:
            return 0.0
        
        prices = [d['low'] for d in historical_data if 'low' in d]
        return min(prices) if prices else 0.0


class IndicatorCalculator:
    """技术指标计算器"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """
        计算 RSI 指标
        
        Args:
            prices: 价格列表（从旧到新）
            period: 周期，默认14
            
        Returns:
            RSI 值 (0-100)
        """
        if len(prices) < period + 1:
            return None
        
        # 计算价格变化
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # 分离涨跌
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # 计算平均涨跌
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    @staticmethod
    def calculate_drawdown(current_price: float, high_52w: float) -> float:
        """
        计算回撤幅度
        
        Args:
            current_price: 当前价格
            high_52w: 52周最高价
            
        Returns:
            回撤百分比
        """
        if high_52w == 0:
            return 0.0
        
        drawdown = ((high_52w - current_price) / high_52w) * 100
        return round(drawdown, 2)
    
    @staticmethod
    def calculate_volume_ratio(current_volume: float, avg_volume_60d: float) -> float:
        """
        计算成交量比率
        
        Args:
            current_volume: 当前成交量
            avg_volume_60d: 60日平均成交量
            
        Returns:
            成交量比率
        """
        if avg_volume_60d == 0:
            return 0.0
        
        return round(current_volume / avg_volume_60d, 2)


class DataCache:
    """数据缓存管理"""
    
    def __init__(self, cache_dir: str = './data'):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_daily_data(self, symbol: str, data: Dict):
        """保存每日数据"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{self.cache_dir}/{symbol}_{date_str}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_historical_cache(self, symbol: str, days: int = 365) -> List[Dict]:
        """从缓存加载历史数据"""
        import os
        import glob
        
        pattern = f"{self.cache_dir}/{symbol}_*.json"
        files = sorted(glob.glob(pattern))[-days:]
        
        historical_data = []
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    historical_data.append(data)
            except:
                continue
        
        return historical_data
    
    def get_52w_high_low(self, symbol: str) -> Dict:
        """获取52周高低点（从缓存）"""
        historical = self.load_historical_cache(symbol, 365)
        
        if not historical:
            return {'high': 0, 'low': 0}
        
        highs = [d.get('high', d.get('current_price', 0)) for d in historical]
        lows = [d.get('low', d.get('current_price', 0)) for d in historical]
        
        return {
            'high': max(highs) if highs else 0,
            'low': min(lows) if lows else 0
        }


if __name__ == '__main__':
    # 测试代码
    api = TencentFinanceAPI()
    
    print("测试腾讯财经接口...")
    test_symbols = ['sh600519', 'hk00700', 'sz300750']
    
    for symbol in test_symbols:
        print(f"\n查询: {symbol}")
        result = api.fetch_realtime(symbol)
        if result['success']:
            print(f"  ✅ {result['name']}: ¥{result['current_price']}")
        else:
            print(f"  ❌ 失败: {result['error']}")
