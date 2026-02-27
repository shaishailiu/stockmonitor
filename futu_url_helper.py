#!/usr/bin/env python3
"""
生成富途牛牛股票链接工具
支持港股、美股、A股等不同市场的代码转换
"""

def get_futu_url(symbol, name):
    """
    根据股票代码生成富途牛牛链接
    
    Args:
        symbol: 股票代码（如 00700.HK, AAPL, 600519.SH）
        name: 股票名称
        
    Returns:
        富途牛牛URL
    """
    # 港股
    if '.HK' in symbol:
        code = symbol.replace('.HK', '')
        return f"https://www.futunn.com/stock/{code}-HK"
    
    # A股 - 上海
    elif '.SH' in symbol:
        code = symbol.replace('.SH', '')
        return f"https://www.futunn.com/stock/{code}-SH"
    
    # A股 - 深圳
    elif '.SZ' in symbol:
        code = symbol.replace('.SZ', '')
        return f"https://www.futunn.com/stock/{code}-SZ"
    
    # 韩国股票
    elif '.KS' in symbol:
        code = symbol.replace('.KS', '')
        return f"https://www.futunn.com/stock/{code}-KS"
    
    # 美股
    elif symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
                     'TSLA', 'NVDA', 'TSM', 'AMD', 'AVGO', 'QCOM', 'INTC', 
                     'MU', 'WDC', 'PDD', 'ORCL', 'SNOW', 'PLTR', 'MDB', 
                     'PFE', 'JNJ', 'KO', 'BRK-B', 'JPM']:
        return f"https://www.futunn.com/stock/{symbol}-US"
    
    # 特殊商品
    elif symbol == 'BTC':
        return "https://www.futunn.com/quote/crypto-detail?code=btcusd&market=us"
    
    elif symbol == 'GOLD':
        return "https://www.futunn.com/quote/future-detail?code=GCmain&market=us"
    
    elif symbol == 'CL=F':
        return "https://www.futunn.com/quote/future-detail?code=CLmain&market=us"
    
    # 默认返回雪球链接作为备选
    else:
        # 雪球格式
        if '.HK' in symbol or '.SH' in symbol or '.SZ' in symbol:
            return f"https://xueqiu.com/S/{symbol.replace('.', '')}"
        else:
            return f"https://xueqiu.com/S/{symbol}"


if __name__ == "__main__":
    # 测试
    test_symbols = [
        ("00700.HK", "腾讯控股"),
        ("AAPL", "苹果"),
        ("600519.SH", "贵州茅台"),
        ("BTC", "比特币"),
        ("GOLD", "黄金")
    ]
    
    for symbol, name in test_symbols:
        url = get_futu_url(symbol, name)
        print(f"{name} ({symbol}): {url}")
