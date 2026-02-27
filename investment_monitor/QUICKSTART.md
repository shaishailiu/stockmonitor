# 快速开始指南

## 🚀 立即测试

### 1. 查看演示（模拟预警场景）
```bash
cd /root/.openclaw/workspace/investment_monitor
python3 demo.py
```

这会展示不同级别的预警是如何触发的。

### 2. 测试真实数据（仅扫描3个资产）
```bash
python3 monitor.py --test --no-notify
```

### 3. 完整扫描（所有17个资产）
```bash
python3 monitor.py --no-notify
```

### 4. 正式运行（扫描 + 推送企业微信）
```bash
python3 monitor.py
```

## ⏰ 定时任务已配置

系统会自动运行，无需手动干预：

- **每日 15:30** - 自动扫描所有资产，触发预警时推送
- **每周日 20:00** - 生成周报汇总

查看定时任务状态：
```bash
openclaw cron list
```

立即触发一次（测试）：
```bash
openclaw cron run 7bacfe2a-be6c-4b9e-915d-cf4a854a66a5
```

## 📊 查看运行日志

```bash
# 查看最新日志
tail -f /root/.openclaw/workspace/investment_monitor/logs/cron_$(date +%Y%m%d).log

# 查看历史日志
ls -lh /root/.openclaw/workspace/investment_monitor/logs/
```

## 🔧 修改配置

编辑配置文件：
```bash
nano /root/.openclaw/workspace/investment_monitor/config.json
```

修改后立即生效，无需重启。

## ❓ 常见操作

### 添加新的监控资产

编辑 `config.json`，在对应的 `a_stock` 或 `hk_stock` 数组中添加：

```json
{
  "symbol": "hk09999",
  "name": "网易",
  "code": "09999",
  "market": "HK"
}
```

### 调整预警阈值

编辑 `config.json` 中的 `alert_thresholds` 部分。

### 暂停/恢复定时任务

```bash
# 暂停
openclaw cron update 7bacfe2a-be6c-4b9e-915d-cf4a854a66a5 --enabled false

# 恢复
openclaw cron update 7bacfe2a-be6c-4b9e-915d-cf4a854a66a5 --enabled true
```

### 查看数据缓存

```bash
ls -lh /root/.openclaw/workspace/investment_monitor/data/
```

每天的价格数据都会保存，用于计算历史指标。

## 🎯 下一步

1. **等待数据积累**：运行2-3周后，52周高点会更准确
2. **观察预警准确性**：看看触发的信号是否靠谱
3. **申请外网白名单**：联系IT部门，解锁美股监控

## 📞 需要帮助？

直接 @OpenClaw，我随时在线！
