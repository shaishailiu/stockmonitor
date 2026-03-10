# -*- coding: utf-8 -*-
import sys
import traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from stock_monitor import _load_pe_data, lookup_pe, get_pe_path

    tests = [
        ("美团 低谷 2025-01-13", {"market": "hk", "symbol": "03690"}, "2025-01-13"),
        ("复星国际 高点 2025-08-12", {"market": "hk", "symbol": "00656"}, "2025-08-12"),
        ("理想汽车 低谷 2024-06-24", {"market": "hk", "symbol": "02015"}, "2024-06-24"),
    ]

    for label, item, date in tests:
        pe_map = _load_pe_data(get_pe_path(item))
        pe = lookup_pe(pe_map, date)
        print(f"{label}: PE = {pe}")

    print()
    print("=" * 60)
    print("重新完整分析 4 只股票:")
    print("=" * 60)

    from stock_monitor import analyze, format_card

    items = [
        {"name": "03690（美团）", "market": "hk", "symbol": "03690"},
        {"name": "TME（腾讯音乐）", "market": "us", "symbol": "106.TME"},
        {"name": "02015（理想汽车）", "market": "hk", "symbol": "02015"},
        {"name": "00656（复星国际）", "market": "hk", "symbol": "00656"},
    ]

    for item in items:
        print(f"\n--- {item['name']} ---")
        result = analyze(item)
        if result:
            print(f"  当前 PE: {result['current_pe']}")
            print(f"  高点 PE ({result['peak_date']}): {result['peak_pe']}")
            td = result.get('trough_date', 'N/A')
            print(f"  低谷 PE ({td}): {result['trough_pe']}")
        else:
            print(f"  未触发预警")

    print("\n[OK] 完成")

except Exception:
    traceback.print_exc()
