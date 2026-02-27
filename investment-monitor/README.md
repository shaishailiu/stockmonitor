# 💰 长线投资底部监控系统

基于 AkShare 的自动化投资机会监控工具

## 📦 快速开始

### 1. 环境配置

```bash
cd investment-monitor
./setup.sh
```

这会自动安装：
- pip (如果没有)
- AkShare
- Pandas
- NumPy
- Requests

### 2. 手动测试

```bash
python3 monitor.py
```

第一次运行会扫描所有配置的资产，可能需要 5-10 分钟（取决于网络）。

### 3. 配置定时任务

监控系统已经为你配置了 cron 任务：
- **每日 15:30** 执行全量扫描
- **每周日 20:00** 生成周度汇总

查看任务状态：
```bash
crontab -l | grep monitor
```

## 🎯 监控资产

### 加密货币
- BTC (比特币)

### 贵金属
- XAU (黄金)

### 美股科技巨头 (Big 7)
- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

### 半导体
- AMD, TSM, AVGO, QCOM, ASML, INTC
- 0981.HK (中芯国际), 1347.HK (华虹半导体)

### 中概股
- 0700.HK (腾讯), BABA, JD, PDD, 3690.HK (美团)
- NTES, BIDU, TCOM, BILI, TME
- XPEV, NIO, LI
- FUTU, TIGR, BEKE, MNSO, EDU

### 港股科技
- 1810.HK (小米), 1024.HK (快手), 0772.HK (阅文), 1211.HK (比亚迪)

### A股龙头
- 600519 (贵州茅台), 300750 (宁德时代), 002594 (比亚迪)
- 600036 (招商银行), 601318 (中国平安)

## 🚨 预警级别

### 🟡 黄色预警（进入观察）
- 回撤≥30% 且 RSI(14)<35
- 触发动作：首次推送，后续每周汇总

### 🟠 橙色预警（持续关注）
- 回撤≥40% 且 RSI(14)<30 且 周线RSI<35
- 回撤≥40% 且 成交量<60日均量的50%
- 触发动作：每日推送

### 🔴 红色预警（重点关注）
- 回撤≥50% 且 周线RSI<30
- 回撤≥50% （极端回撤）
- 触发动作：实时推送 + 语音提醒

## 📊 监控指标

### 基础数据（所有资产）
- 当前价格
- 52周最高价
- 回撤幅度

### 深度指标（回撤≥30%时）
- RSI(14) 日线值
- RSI 周线值
- 成交量 vs 60日均量
- PE/PB 估值及历史分位（待实现）
- MACD 状态（待实现）

## 📂 文件说明

```
investment-monitor/
├── monitor.py              # 主监控脚本
├── requirements.txt        # Python 依赖
├── setup.sh               # 环境配置脚本
├── README.md              # 本文档
├── monitor_data.json      # 运行时数据（自动生成）
└── alert_history.json     # 预警历史（自动生成）
```

## 🔧 高级用法

### 单资产分析（待实现）

```bash
python3 monitor.py analyze AAPL
python3 monitor.py analyze 0700.HK
python3 monitor.py analyze 600519
```

### 查看历史预警

```bash
cat alert_history.json | python3 -m json.tool
```

### 自定义扫描时间

编辑 crontab：
```bash
crontab -e
```

## 📝 投资策略（记住的偏好）

1. ✅ 长线投资者，关注 3-5 年周期
2. ✅ 优先龙头公司，不考虑小市值
3. ✅ 对中概股有较高风险偏好
4. ✅ BTC 采用减半周期策略
5. ✅ 极度恐慌时加大关注
6. ✅ 分批建仓：30%跌→10%仓，40%跌→20%仓，50%跌→30%仓

## ⚠️ 注意事项

1. **数据延迟**：AkShare 获取的数据可能有 15 分钟延迟
2. **API 限制**：频繁请求可能被限流，建议每日扫描不超过 3 次
3. **网络要求**：需要稳定的网络连接
4. **仅供参考**：所有预警仅供参考，不构成投资建议

## 🐛 故障排查

### 获取数据失败

```bash
# 测试 AkShare 是否正常
python3 -c "import akshare as ak; print(ak.stock_us_spot_em().head())"
```

### 定时任务未执行

```bash
# 检查 cron 日志
grep CRON /var/log/syslog | tail -20
```

### 重新安装

```bash
rm -rf ~/.local/lib/python3.*/site-packages/akshare*
./setup.sh
```

## 📈 未来计划

- [ ] 估值分位数计算（PE/PB 5年历史）
- [ ] MACD 底背离检测
- [ ] 基本面排雷（营收/现金流/负债率）
- [ ] VIX 恐慌指数
- [ ] 北向资金监控
- [ ] 语音播报（TTS）
- [ ] 周度汇总报告
- [ ] 单资产深度分析
- [ ] Web 控制面板

## 💡 贡献

如果发现 bug 或有改进建议，请创建 issue 或直接修改代码。

---

**免责声明**：本工具仅用于学习和研究，不构成任何投资建议。投资有风险，入市需谨慎。
