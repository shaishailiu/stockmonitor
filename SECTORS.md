# 板块分类说明

## 所有板块（sector 值）

| sector 值 | 中文名 | 说明 |
|-----------|--------|------|
| `gaming` | 游戏 | 游戏行业相关公司 |
| `tech` | 科技 | 互联网、软件、平台型科技公司 |
| `chip_ai` | 芯片/AI | 半导体、AI 芯片、算力相关 |
| `consumer` | 消费 | 消费品、零售、餐饮、日用品 |
| `finance` | 金融 | 银行、保险、券商 |
| `healthcare` | 医疗 | 医药、生物科技、医疗器械 |
| `resources_industrial` | 资源/工业 | 矿业、能源、制造业 |
| `auto_energy` | 汽车/新能源 | 新能源汽车、电池、充电桩 |

## 快捷分组

`--sector` / `-s` 参数支持以下快捷名（大小写不敏感）：

| 快捷名 | 包含的 sector | 说明 |
|--------|--------------|------|
| `G` | gaming | 🎮 游戏行业 |
| `T` | tech, chip_ai | 💻 科技行业 |
| `O` | 除 gaming/tech/chip_ai 外所有 | 📊 其他行业 |
| `all` | 全部 | 不过滤 |

## 使用示例

```bash
# 使用快捷名
python -X utf8 stock_monitor_v2.py -s G -o r.txt       # 🎮 游戏行业
python -X utf8 stock_monitor_v2.py -s T -o r.txt       # 💻 科技行业
python -X utf8 stock_monitor_v2.py -s O -o r.txt       # 📊 其他行业
python -X utf8 stock_monitor_v2.py -s all -o r.txt     # 全部

# 自定义组合（逗号分隔 sector 值）
python -X utf8 stock_monitor_v2.py -s consumer,healthcare -o r.txt
python -X utf8 stock_monitor_v2.py -s chip_ai -o r.txt

# 排除模式
python -X utf8 stock_monitor_v2.py -e gaming -o r.txt  # 全部但排除游戏
```
