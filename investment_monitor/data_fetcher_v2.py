#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合数据获取模块 - 腾讯财经 + Alpha Vantage
"""
import urllib.request
import json
import re
from datetime import datetime
from typing import Dict, Optional, List
import time

# Alpha Vantage API Key
ALPHA_VANTAGE_KEY = 'RRWQBOOV7VEQZS1T'


class TencentFinanceAPI:
    """腾讯财经数据接口 - 用于A股和港股"""
    
    BASE_URL = "http://qt.gtimg.cn/q="
    HISTORY_URL = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    
    @staticmethod
    def fetch_realtime(symbol: str) -> Optional[Dict]:
        """获取实时行情数据"""
        try:
            url = f"{TencentFinanceAPI.BASE_URL}{symbol}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = response.read().decode('gbk')
            
            match = re.search(r'"(.+)"', data)
            if not match:
                return {'success': False, 'error': '数据格式错误'}
            
            fields = match.group(1).split('~')
            if len(fields) < 10 or not fields[3] or fields[3] == '0':
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
    def fetch_history_a_stock(symbol: str, days: int = 365) -> Optional[Dict]:
        """
        获取A股历史数据（完整52周）
        symbol: sh600519, sz000001
        """
        try:
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f'{TencentFinanceAPI.HISTORY_URL}?param={symbol},day,{start_date},{end_date},500,qfq'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'http://gu.qq.com/'})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            if result['code'] != 0:
                return {'success': False, 'error': f"API返回错误: {result.get('msg')}"}
            
            kline_data = result['data'][symbol]
            if not kline_data or 'qfqday' not in kline_data:
                return {'success': False, 'error': '无K线数据'}
            
            daily_data = kline_data['qfqday']
            if not daily_data:
                return {'success': False, 'error': 'K线数据为空'}
            
            # 解析数据 [日期, 开盘, 收盘, 最高, 最低, 成交量]
            prices = []
            for day in daily_data:
                prices.append({
                    'date': day[0],
                    'open': float(day[1]),
                    'close': float(day[2]),
                    'high': float(day[3]),
                    'low': float(day[4]),
                    'volume': float(day[5]) if len(day) > 5 else 0
                })
            
            if not prices:
                return {'success': False, 'error': '无法解析价格数据'}
            
            # 计算关键指标
            current_price = prices[-1]['close']
            high_52w = max(p['high'] for p in prices)
            low_52w = min(p['low'] for p in prices)
            high_record = max(prices, key=lambda x: x['high'])
            
            return {
                'success': True,
                'current_price': current_price,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'high_date': high_record['date'],
                'days': len(prices),
                'prices': prices
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def fetch_history_hk_stock(symbol: str, days: int = 365) -> Optional[Dict]:
        """
        获取港股历史数据（完整52周）
        symbol: hk00700, hk03690
        """
        try:
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f'{TencentFinanceAPI.HISTORY_URL}?param={symbol},day,{start_date},{end_date},500,'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'http://gu.qq.com/'})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            if result['code'] != 0:
                return {'success': False, 'error': f"API返回错误: {result.get('msg')}"}
            
            kline_data = result['data'][symbol]
            if not kline_data or 'day' not in kline_data:
                return {'success': False, 'error': '无K线数据'}
            
            daily_data = kline_data['day']
            if not daily_data:
                return {'success': False, 'error': 'K线数据为空'}
            
            # 解析数据 [日期, 开盘, 收盘, 最高, 最低, 成交量]
            prices = []
            for day in daily_data:
                prices.append({
                    'date': day[0],
                    'open': float(day[1]),
                    'close': float(day[2]),
                    'high': float(day[3]),
                    'low': float(day[4]),
                    'volume': float(day[5]) if len(day) > 5 else 0
                })
            
            if not prices:
                return {'success': False, 'error': '无法解析价格数据'}
            
            # 计算关键指标
            current_price = prices[-1]['close']
            high_52w = max(p['high'] for p in prices)
            low_52w = min(p['low'] for p in prices)
            high_record = max(prices, key=lambda x: x['high'])
            
            return {
                'success': True,
                'current_price': current_price,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'high_date': high_record['date'],
                'days': len(prices),
                'prices': prices
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class AlphaVantageAPI:
    """Alpha Vantage API - 用于美股"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    @staticmethod
    def fetch_daily(symbol: str, compact: bool = True) -> Optional[Dict]:
        """
        获取美股日线数据
        compact=True: 最近100天（免费）
        compact=False: 完整历史（需付费）
        """
        try:
            outputsize = 'compact' if compact else 'full'
            url = f"{AlphaVantageAPI.BASE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_KEY}"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if 'Error Message' in data:
                return {'success': False, 'error': data['Error Message']}
            
            if 'Note' in data:  # API限流
                return {'success': False, 'error': 'API限流，请稍后重试'}
            
            if 'Time Series (Daily)' not in data:
                return {'success': False, 'error': '无时间序列数据'}
            
            time_series = data['Time Series (Daily)']
            
            # 转换为统一格式
            prices = []
            for date, values in sorted(time_series.items(), reverse=True)[:365]:
                prices.append({
                    'date': date,
                    'open': float(values['1. open']),
                    'close': float(values['4. close']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'volume': float(values['5. volume'])
                })
            
            if not prices:
                return {'success': False, 'error': '无价格数据'}
            
            current_price = prices[0]['close']
            high_52w = max(p['high'] for p in prices[:min(252, len(prices))])  # 252个交易日≈1年
            low_52w = min(p['low'] for p in prices[:min(252, len(prices))])
            high_record = max(prices[:min(252, len(prices))], key=lambda x: x['high'])
            
            return {
                'success': True,
                'current_price': current_price,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'high_date': high_record['date'],
                'days': len(prices),
                'prices': prices
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class HybridDataFetcher:
    """混合数据获取器 - 自动选择最佳数据源"""
    
    def __init__(self):
        self.tencent = TencentFinanceAPI()
        self.alpha = AlphaVantageAPI()
        self.alpha_call_count = 0
        self.alpha_last_call = 0
    
    def fetch_stock_data(self, symbol: str, market: str) -> Optional[Dict]:
        """
        根据市场自动选择数据源
        market: A, HK, US
        """
        if market == 'A':
            # A股：使用腾讯财经历史数据
            return self._fetch_a_stock(symbol)
        elif market == 'HK':
            # 港股：使用腾讯实时数据
            return self._fetch_hk_stock(symbol)
        elif market == 'US':
            # 美股：使用Alpha Vantage
            return self._fetch_us_stock(symbol)
        else:
            return {'success': False, 'error': f'不支持的市场: {market}'}
    
    def _fetch_a_stock(self, symbol: str) -> Optional[Dict]:
        """获取A股数据（完整历史）"""
        result = self.tencent.fetch_history_a_stock(symbol, 365)
        if result['success']:
            result['source'] = 'Tencent'
            result['market'] = 'A'
        return result
    
    def _fetch_hk_stock(self, symbol: str) -> Optional[Dict]:
        """获取港股数据（现在也有完整历史了！）"""
        result = self.tencent.fetch_history_hk_stock(symbol, 365)
        if result['success']:
            result['source'] = 'Tencent'
            result['market'] = 'HK'
        return result
    
    def _fetch_us_stock(self, symbol: str) -> Optional[Dict]:
        """获取美股数据（Alpha Vantage，有限流）"""
        # API限流控制：每分钟5次
        current_time = time.time()
        if current_time - self.alpha_last_call < 12:  # 12秒间隔
            time.sleep(12 - (current_time - self.alpha_last_call))
        
        result = self.alpha.fetch_daily(symbol, compact=True)
        if result['success']:
            result['source'] = 'AlphaVantage'
            result['market'] = 'US'
        
        self.alpha_last_call = time.time()
        self.alpha_call_count += 1
        
        return result


class IndicatorCalculator:
    """技术指标计算器"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """计算RSI"""
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    @staticmethod
    def calculate_drawdown(current_price: float, high_52w: float) -> float:
        """计算回撤"""
        if high_52w == 0:
            return 0.0
        return round(((high_52w - current_price) / high_52w) * 100, 2)


if __name__ == '__main__':
    # 测试
    fetcher = HybridDataFetcher()
    
    print("="*60)
    print("混合数据源测试")
    print("="*60)
    
    test_cases = [
        ('sh600519', 'A', '贵州茅台'),
        ('hk00700', 'HK', '腾讯'),
        ('AAPL', 'US', '苹果'),
    ]
    
    for symbol, market, name in test_cases:
        print(f"\n{name} ({symbol}-{market}):")
        result = fetcher.fetch_stock_data(symbol, market)
        
        if result['success']:
            print(f"  ✅ 来源: {result['source']}")
            print(f"  当前价: {result['current_price']:.2f}")
            print(f"  52周高: {result['high_52w']:.2f} ({result['high_date']})")
            drawdown = IndicatorCalculator.calculate_drawdown(result['current_price'], result['high_52w'])
            print(f"  回撤: {drawdown:.2f}%")
        else:
            print(f"  ❌ {result['error']}")
