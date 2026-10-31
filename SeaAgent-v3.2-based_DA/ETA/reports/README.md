# ETA 实验结果

本目录只保留当前最终结果，不保存旧实验历史：

- `experiments.md`：消融组与调参组的四列结果表；
- `experiments.csv`：同一结果表的 CSV 版本；
- `best_run.md`：最终模型摘要；
- `best_run.csv`：最终模型的四列表格；
- `best_run.json`：最终模型的环境、验证集指标、测试指标和权重摘要；
- `dataset_summary.json`：最终五数组数据集的切分与哈希；
- `latest.json`：完整机器可读实验记录；
- `reproducibility.json`：相同配置独立训练后的指标比较。

最终权重保存在 `artifacts/best_model/best.pt`，其他实验权重、checkpoint、
embedding 缓存和运行日志在发布后删除。
