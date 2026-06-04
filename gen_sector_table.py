"""从 config.json 生成板块分类表格 (Markdown)."""
import json, os

STOCK_TYPE_CN = {
    "cyclical": "周期型",
    "non_cyclical": "非周期",
    "unclassified": "未分类",
}

MARKET_CN = {
    "us": "美股",
    "hk": "港股",
    "cn": "A股",
    "a": "A股",
}

SECTOR_CN = {
    "gaming": "🎮 游戏",
    "tech": "💻 科技",
    "chip_ai": "🔬 芯片/AI",
    "consumer": "🛒 消费",
    "finance": "🏦 金融",
    "healthcare": "💊 医疗",
    "resources_industrial": "🏭 资源/工业",
    "auto_energy": "🚗 汽车/新能源",
}


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stocks = data["stocks"]

    # 按 sector 分组
    sectors: dict[str, list[dict]] = {}
    for s in stocks:
        sec = s.get("sector", "unknown")
        sectors.setdefault(sec, []).append(s)

    buf: list[str] = []
    buf.append("# 股票板块分类一览\n")
    buf.append(f"共 **{len(stocks)}** 只股票，**{len(sectors)}** 个板块\n")

    # 按预设顺序排列
    order = ["gaming", "tech", "chip_ai", "consumer", "healthcare", "finance", "resources_industrial", "auto_energy"]
    for sec in order:
        if sec not in sectors:
            continue
        items = sectors[sec]
        label = SECTOR_CN.get(sec, f"📋 {sec}")
        buf.append(f"## {label}（{len(items)}只）\n")
        buf.append("| 股票 | 市场 | 类型 |")
        buf.append("|------|------|------|")
        for s in sorted(items, key=lambda x: x["name"]):
            mkt = MARKET_CN.get(s["market"], s["market"])
            stype = STOCK_TYPE_CN.get(s["stock_type"], s["stock_type"])
            buf.append(f"| {s['name']} | {mkt} | {stype} |")
        buf.append("")

    # 兜底：未知 sector
    for sec, items in sectors.items():
        if sec in order:
            continue
        buf.append(f"## 📋 {sec}（{len(items)}只）\n")
        buf.append("| 股票 | 市场 | 类型 |")
        buf.append("|------|------|------|")
        for s in sorted(items, key=lambda x: x["name"]):
            mkt = MARKET_CN.get(s["market"], s["market"])
            stype = STOCK_TYPE_CN.get(s["stock_type"], s["stock_type"])
            buf.append(f"| {s['name']} | {mkt} | {stype} |")
        buf.append("")

    output_path = os.path.join(os.path.dirname(__file__), "SECTOR_TABLE.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf))
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
