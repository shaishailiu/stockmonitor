#!/usr/bin/env python3
"""
自动更新热门股票配置
每天晚上8点执行：
1. 获取港股、A股、美股 TOP 50 热度排名（按成交额）
2. 对比 config.json 中的股票列表
3. 自动添加热度高但不在配置中的股票
"""

import json
import akshare as ak
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

# 配置
CONFIG_FILE = Path(__file__).parent / "config.json"
TOP_N = 50  # 获取前50名
MIN_VOLUME_THRESHOLD = {
    'hk': 5e8,    # 港股最低5亿港币成交额
    'a': 10e8,    # A股最低10亿人民币成交额
    'us': 1e8     # 美股最低1亿美元成交额（暂时不可用，后续添加）
}

def log(msg):
    """打印日志"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_config():
    """加载配置文件"""
    if not CONFIG_FILE.exists():
        log(f"❌ 配置文件不存在: {CONFIG_FILE}")
        return None
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config):
    """保存配置文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    log(f"✅ 配置已保存到: {CONFIG_FILE}")

def get_hk_hot_stocks(top_n=50):
    """获取港股热度 TOP N（按成交额）"""
    log("📊 正在获取港股热度排行...")
    try:
        df = ak.stock_hk_spot()
        
        if df.empty:
            log("❌ 港股数据为空")
            return []
        
        # 清理成交额列
        df['成交额_数值'] = pd.to_numeric(df['成交额'], errors='coerce')
        
        # 过滤掉成交额过低的股票
        df = df[df['成交额_数值'] >= MIN_VOLUME_THRESHOLD['hk']]
        
        # 按成交额排序
        df_sorted = df.sort_values(by='成交额_数值', ascending=False).head(top_n)
        
        # 提取股票信息
        stocks = []
        for _, row in df_sorted.iterrows():
            code = str(row['symbol']) if 'symbol' in row else str(row.get('代码', ''))
            name = str(row['name']) if 'name' in row else str(row.get('中文名称', ''))
            volume = row['成交额_数值']
            
            # 格式化代码（确保5位数）
            code = code.zfill(5)
            
            stocks.append({
                'symbol': code,
                'name': name,
                'market': 'hk',
                'volume': volume
            })
        
        log(f"✅ 成功获取 {len(stocks)} 只港股")
        return stocks
        
    except Exception as e:
        log(f"❌ 获取港股数据失败: {e}")
        return []

def get_a_hot_stocks(top_n=50):
    """获取A股热度 TOP N（按成交额）- 使用新浪财经接口"""
    log("📊 正在获取A股热度排行...")
    
    # 方法1: 新浪财经接口（主要方法）
    try:
        log("  🔄 尝试新浪财经接口...")
        url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
        params = {
            "page": 1,
            "num": top_n,
            "sort": "amount",  # 按成交额排序
            "asc": 0,          # 降序
            "node": "hs_a",    # 沪深A股
            "symbol": "",
            "_s_r_a": "page"
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if not data:
                log("  ❌ 新浪财经返回数据为空")
            else:
                # 提取股票信息
                stocks = []
                for item in data:
                    code = str(item.get('code', ''))
                    name = str(item.get('name', ''))
                    volume = float(item.get('amount', 0))  # 成交额（元）
                    
                    # 过滤成交额过低的股票
                    if volume >= MIN_VOLUME_THRESHOLD['a']:
                        stocks.append({
                            'symbol': code,
                            'name': name,
                            'market': 'a',
                            'volume': volume
                        })
                
                log(f"  ✅ 新浪财经成功获取 {len(stocks)} 只A股")
                return stocks
        else:
            log(f"  ❌ 新浪财经接口失败: HTTP {response.status_code}")
            
    except Exception as e:
        log(f"  ❌ 新浪财经接口异常: {e}")
    
    # 方法2: AKShare 备用方法
    try:
        log("  🔄 尝试 AKShare 备用接口...")
        df_sh = ak.stock_zh_a_spot_em()
        
        if df_sh.empty:
            log("  ❌ AKShare A股数据为空")
            return []
        
        # 清理成交额列（单位：元）
        if '成交额' in df_sh.columns:
            df_sh['成交额_数值'] = pd.to_numeric(df_sh['成交额'], errors='coerce')
        else:
            log("  ❌ AKShare A股数据缺少成交额列")
            return []
        
        # 过滤掉成交额过低的股票
        df_sh = df_sh[df_sh['成交额_数值'] >= MIN_VOLUME_THRESHOLD['a']]
        
        # 按成交额排序
        df_sorted = df_sh.sort_values(by='成交额_数值', ascending=False).head(top_n)
        
        # 提取股票信息
        stocks = []
        for _, row in df_sorted.iterrows():
            code = str(row['代码'])
            name = str(row['名称'])
            volume = row['成交额_数值']
            
            stocks.append({
                'symbol': code,
                'name': name,
                'market': 'a',
                'volume': volume
            })
        
        log(f"  ✅ AKShare 成功获取 {len(stocks)} 只A股")
        return stocks
        
    except Exception as e:
        log(f"  ❌ AKShare 接口失败: {e}")
        return []

def get_us_hot_stocks(top_n=50):
    """获取美股热度 TOP N（暂时不可用）"""
    log("⚠️  美股热度数据暂不支持，跳过...")
    return []

def merge_hot_stocks_to_config(config, hot_stocks):
    """将热门股票合并到配置中"""
    if not config or 'stocks' not in config:
        log("❌ 配置文件格式错误")
        return 0
    
    existing_symbols = {stock['symbol'] for stock in config['stocks']}
    added_count = 0
    
    for stock in hot_stocks:
        symbol = stock['symbol']
        market = stock['market']
        name = stock['name']
        
        if symbol not in existing_symbols:
            # 构造新条目
            new_entry = {
                'name': f"{symbol}（{name}）",
                'market': market,
                'symbol': symbol
            }
            
            config['stocks'].append(new_entry)
            existing_symbols.add(symbol)
            added_count += 1
            log(f"  ➕ 新增: {market.upper()} {symbol} {name} (成交额: {stock['volume']/1e8:.2f}亿)")
    
    return added_count

def main():
    """主函数"""
    log("=" * 100)
    log("🚀 开始自动更新热门股票配置")
    log("=" * 100)
    
    # 1. 加载配置
    config = load_config()
    if config is None:
        return
    
    log(f"📋 当前配置中有 {len(config['stocks'])} 只股票")
    
    # 2. 获取各市场热度排行
    all_hot_stocks = []
    
    # 港股
    hk_stocks = get_hk_hot_stocks(TOP_N)
    all_hot_stocks.extend(hk_stocks)
    
    # A股
    a_stocks = get_a_hot_stocks(TOP_N)
    all_hot_stocks.extend(a_stocks)
    
    # 美股（暂时不可用）
    us_stocks = get_us_hot_stocks(TOP_N)
    all_hot_stocks.extend(us_stocks)
    
    log(f"\n📊 总共获取到 {len(all_hot_stocks)} 只热门股票")
    
    # 3. 合并到配置
    log("\n🔄 开始合并新股票到配置...")
    added_count = merge_hot_stocks_to_config(config, all_hot_stocks)
    
    # 4. 保存配置
    if added_count > 0:
        save_config(config)
        log(f"\n✅ 成功添加 {added_count} 只新股票！")
        log(f"📊 配置中现在共有 {len(config['stocks'])} 只股票")
    else:
        log("\n✅ 所有热门股票已在配置中，无需添加")
    
    log("=" * 100)
    log("✅ 任务完成！")
    log("=" * 100)

if __name__ == "__main__":
    main()
