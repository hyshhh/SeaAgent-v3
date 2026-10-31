# ETA 消融与调参结果

最优模型：`T3 Unweighted-Both`

| 实验设置 | 消融组/调参组 | 双头精度 | 推理速度（单次ms） |
|---|---|---|---:|
| A1 MLP-Cur | 消融组 | AH 90.36% / CH 92.77% | 0.201 |
| A2 MLP-Hist | 消融组 | AH 90.36% / CH 93.37% | 0.351 |
| A3 SA-DH | 消融组 | AH 90.36% / CH 93.98% | 1.036 |
| A4 CA-only | 消融组 | AH 92.17% / CH 94.58% | 3.529 |
| A5 CA-SH | 消融组 | AH 90.36% / CH — | 4.742 |
| A6 Full-ETA | 消融组 | AH 90.36% / CH 89.16% | 4.669 |
| T1 Balanced-Baseline | 调参组 | AH 90.36% / CH 89.16% | 4.985 |
| T2 AH-Unweighted | 调参组 | AH 90.36% / CH 89.76% | 4.976 |
| T3 Unweighted-Both | 调参组 | AH 90.36% / CH 93.98% | 4.802 |
| T4 Small-High-Reg | 调参组 | AH 90.36% / CH 94.58% | 4.809 |
| T5 Base-High-Reg | 调参组 | AH 90.36% / CH 88.55% | 4.944 |
| T6 Low-LR | 调参组 | AH 90.36% / CH 92.17% | 4.590 |

消融组说明：

- A1 MLP-Cur：仅使用当前候选 embedding。
- A2 MLP-Hist：当前 embedding 与 3 帧历史均值拼接。
- A3 SA-DH：当前帧与历史 token 拼接后做自注意力。
- A4 CA-only：保留交叉注意力，移除残差和 FFN 融合。
- A5 CA-SH：保留完整交叉注意力，仅保留 AH 单头，因此 CH 不适用。
- A6 Full-ETA：完整串行交叉注意力与 AH/CH 双头。

说明：双头精度为固定测试集结果；模型选择仅使用验证集平均双头准确率，同分时以 batch-1 P50 时延决胜。推理速度是 batch-1 单次前向的 P50，不包含图像编码。准确率受类别不平衡影响，精确率、召回率、F1、校准与混淆矩阵保存在latest.json；本表不据此宣称均衡分类性能。
