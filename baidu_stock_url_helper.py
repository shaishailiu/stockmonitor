#!/usr/bin/env python3
"""
生成百度股市通链接工具（修正版）
支持港股、美股、A股等不同市场的代码转换
"""

def get_baidu_stock_url(symbol, name):
    """
    根据股票代码生成百度股市通链接
    
    正确格式：
    - 港股: hk-{代码}
    - A股: ab-{代码}
    - 美股: us-{大写代码}
    
    Args:
        symbol: 股票代码（如 00700.HK, AAPL, 600519.SH）
        name: 股票名称
        
    Returns:
        百度股市通URL
    """
    # 港股
    if '.HK' in symbol:
        code = symbol.replace('.HK', '')
        return f"https://gushitong.baidu.com/stock/hk-{code}"
    
    # A股 - 统一使用 ab- 前缀（包括上海和深圳）
    elif '.SH' in symbol or '.SZ' in symbol:
        code = symbol.replace('.SH', '').replace('.SZ', '')
        return f"https://gushitong.baidu.com/stock/ab-{code}"
    
    # A股纯数字代码
    elif symbol.isdigit() and len(symbol) == 6:
        return f"https://gushitong.baidu.com/stock/ab-{symbol}"
    
    # 美股 - 使用 us- 前缀，保持大写
    elif symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
                     'TSLA', 'NVDA', 'TSM', 'AMD', 'AVGO', 'QCOM', 'INTC', 
                     'MU', 'WDC', 'PDD', 'ORCL', 'SNOW', 'PLTR', 'MDB', 
                     'PFE', 'JNJ', 'KO', 'BRK-B', 'JPM']:
        return f"https://gushitong.baidu.com/stock/us-{symbol}"
    
    # 韩国股票 (使用雪球)
    elif '.KS' in symbol:
        return f"https://xueqiu.com/S/{symbol.replace('.', '')}"
    
    # 特殊商品 - 使用新浪财经
    elif symbol == 'BTC':
        return "https://finance.sina.com.cn/blockchain/coin/btc.html"
    
    elif symbol == 'GOLD':
        return "https://finance.sina.com.cn/money/future/quote/GC.html"
    
    elif symbol == 'CL=F':
        return "https://finance.sina.com.cn/money/future/quote/CL.html"
    
    # 默认返回雪球链接作为备选
    else:
        if '.HK' in symbol or '.SH' in symbol or '.SZ' in symbol:
            return f"https://xueqiu.com/S/{symbol.replace('.', '')}"
        else:
            return f"https://xueqiu.com/S/{symbol}"


def format_markdown_link(name, symbol):
    """
    生成Markdown格式的链接
    
    Args:
        name: 股票名称
        symbol: 股票代码
        
    Returns:
        Markdown格式链接
    """
    url = get_baidu_stock_url(symbol, name)
    return f"[{name}]({url})"


if __name__ == "__main__":
    # 测试
    test_symbols = [
        ("腾讯控股", "00700.HK"),
        ("苹果", "AAPL"),
        ("贵州茅台", "600519.SH"),
        ("宁德时代", "300750.SZ"),
        ("比亚迪", "002594.SZ"),
        ("美光科技", "MU"),
        ("比特币", "BTC")
    ]
    
    print("百度股市通链接测试（修正版）：\n")
    for name, symbol in test_symbols:
        url = get_baidu_stock_url(symbol, name)
        markdown = format_markdown_link(name, symbol)
        print(f"{name} ({symbol}):")
        print(f"  URL: {url}")
        print(f"  Markdown: {markdown}")
        print()
