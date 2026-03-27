"""test_resilience.py — 管理层韧性评估 单只测试工具

用法：
  python test_resilience.py 00700          # 测试腾讯控股
  python test_resilience.py 106.TME        # 测试腾讯音乐
  python test_resilience.py 002594         # 测试比亚迪
  python test_resilience.py 00700 --no-llm # 只看V型周期检测，不调API
"""

import json
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

from management_resilience import (
    load_config, load_stock_data, find_v_cycles, extract_features,
    build_prompt, call_deepseek, parse_llm_response
)


def find_stock(symbol: str) -> dict | None:
    """按 symbol 或名称关键词查找股票"""
    stocks = load_config()
    # 精确匹配 symbol
    for s in stocks:
        if s["symbol"] == symbol:
            return s
    # 模糊匹配名称
    for s in stocks:
        if symbol.lower() in s["name"].lower() or symbol.lower() in s["symbol"].lower():
            return s
    return None


def test_stock(symbol: str, call_llm: bool = True):
    item = find_stock(symbol)
    if not item:
        print(f"未找到股票: {symbol}")
        print("提示: 可用 symbol（如 00700）或名称关键词（如 腾讯）")
        return

    print(f"{'='*70}")
    print(f"  {item['name']}  ({item['symbol']}, {item['market']})")
    print(f"{'='*70}")

    # 加载数据
    df = load_stock_data(item)
    if df is None:
        print("数据不足或不存在")
        return
    print(f"数据: {len(df)} 条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    # V型周期检测
    cycles = find_v_cycles(df)
    if not cycles:
        print("未检测到完整的V型周期（跌>=30% 且 恢复>=50%）")
        return

    print(f"\n检测到 {len(cycles)} 个V型周期:")
    for i, c in enumerate(cycles, 1):
        print(f"\n  --- 周期 {i} ---")
        print(f"  峰值: {c['peak_price']}  ({c['peak_date']})")
        print(f"  谷底: {c['trough_price']}  ({c['trough_date']})  下跌 {c['drop_pct']}%, {c['drop_days']} 个交易日")
        print(f"  恢复: {c['recovery_price']}  ({c['recovery_date']})  恢复到峰值 {c['recovery_pct']}%, {c['recovery_days']} 个交易日")

    # 特征提取
    features_list = extract_features(item, df)
    if not features_list:
        print("\n特征提取失败")
        return

    for feat in features_list:
        print(f"\n{'='*70}")
        print(f"  周期 {feat['cycle_index']} 详情")
        print(f"{'='*70}")
        print(f"季度股价: {feat['quarterly_prices']}")
        print(f"月线股价: {feat['monthly_prices']}")

        if not call_llm:
            print("\n(跳过 LLM 调用, 使用 --no-llm 参数)")
            continue

        # 构造 prompt
        prompt = build_prompt(feat)
        print(f"\nPrompt: {len(prompt)} 字符")
        print("调用腾讯云 DeepSeek API (联网搜索)...")

        try:
            raw = call_deepseek(prompt, enable_search=True)
            result = parse_llm_response(raw)

            if result:
                print(f"\n{'─'*70}")
                print(f"  LLM 分析结果")
                print(f"{'─'*70}")

                # 危机类型
                print(f"\n危机类型: {result.get('crisis_type', '?')}")

                # 危机时间线
                timeline = result.get("crisis_timeline", [])
                if timeline:
                    print(f"\n--- 危机时间线 ({len(timeline)} 个事件) ---")
                    for ev in timeline:
                        print(f"  [{ev.get('date','')}] 股价 {ev.get('stock_price','?')}")
                        print(f"    {ev.get('event','')}")

                # 管理层行动
                actions = result.get("management_response_timeline", [])
                if actions:
                    print(f"\n--- 管理层行动 ({len(actions)} 个决策) ---")
                    for act in actions:
                        print(f"  [{act.get('date','')}] 股价 {act.get('stock_price','?')}")
                        print(f"    行动: {act.get('action','')}")
                        print(f"    效果: {act.get('result','')}")
                        print(f"    股价变化: {act.get('stock_price_after','?')}")

                # 总结
                print(f"\n--- 评估 ---")
                print(f"  行动质量: {result.get('action_quality', '?')}")
                print(f"  恢复驱动: {result.get('recovery_driver', '?')}")
                print(f"  韧性评分: {result.get('management_resilience_score', '?')}/100")
                print(f"  关键洞察: {result.get('key_insight', '?')}")
                print(f"  一句话总结: {result.get('one_line_summary', '?')}")

                # 也输出完整 JSON
                print(f"\n--- 完整 JSON ---")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("JSON 解析失败，原始返回:")
                print(raw[:1000])

        except Exception as e:
            print(f"API 调用异常: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        # 列出部分可用股票
        stocks = load_config()
        print("部分可用股票:")
        for s in stocks[:20]:
            print(f"  {s['symbol']:<12} {s['name']}")
        print(f"  ... 共 {len(stocks)} 只")
        return

    symbol = sys.argv[1]
    call_llm = "--no-llm" not in sys.argv

    test_stock(symbol, call_llm)


if __name__ == "__main__":
    main()
