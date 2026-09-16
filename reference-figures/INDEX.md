# reference-figures —— 对比参考图清单

用途：为论文中的 `(a) End-To-End Video Language Model` 图找可引用的对照参考图（视频帧 → Video Tokens → LLM → Answer，外加 User Query 文本输入）。

所有图片已本地化，均取自公开论文（arXiv / 官方页）。**使用前请核对原论文授权协议**（arXiv 多为 perpetual non-exclusive license 或 CC BY，正式发表/再分发前请确认）。

## 文件清单

| 文件 | 来源 | 图号与图注（原文） | 获取方式 / 直链 |
|---|---|---|---|
| `01-videochatgpt-fig1-architecture.png` | Video-ChatGPT, arXiv:2306.05424, p.4 | Fig. 1 "Architecture of Video-ChatGPT. Video-ChatGPT leverages the CLIP-L/14 visual encoder to extract..." | 从论文 PDF 抽内嵌原始位图（4450x2021） |
| `02-videollava-fig2a-framework.png` | Video-LLaVA, arXiv:2311.10122v3, p.5 | Fig. 2(a) "Illustration of Video-LLaVA"（Fig.2 总注：the Video-LLaVA framework demonstrates a data flow that generates corresponding responses based on input instructions） | 300dpi 渲染 p.5 后精确裁剪 |
| `03-videollava-fig2-full.png` | 同上 | Fig. 2 全图（(a) 框架 + (b) 性能对比） | 同上 |
| `05-mallms-survey-fig2-general-architecture.png` | MM-LLMs 综述, arXiv:2401.13601 | Fig. 2 "The general model architecture of MM-LLMs and the implementation choices for each component."（Modality Encoder -> Input Projector -> LLM Backbone -> Output Projector -> Modality Generator） | https://ar5iv.labs.arxiv.org/html/2401.13601/assets/model-final.png |
| `06-watervideoqa-fig1-overview.png` | WaterVideoQA (NaviMind), arXiv:2602.22923 | Fig. 1 "The overview of pipeline, including (a) the proposed WaterVideoQA dataset; (b) the proposed multi-agent neuro-symbolic reasoning system: NaviMind; (c) the real-word application scenarios..." | https://ar5iv.labs.arxiv.org/html/2602.22923/assets/overview.png |
| `07-watervideoqa-fig4-navimind-architecture.png` | 同上 | Fig. 4 "The architecture of NaviMind, a multi-agent reasoning system for waterway navigation. NaviMind has two inputs (a user query and a video clip) and an output answer. It has 5 agents for routing, captioning, reasoning, grading and summary." | https://ar5iv.labs.arxiv.org/html/2602.22923/assets/pipeline.png |
| `08-surveillancevqa-fig2-framework.png` | SurveillanceVQA-589K, arXiv:2505.12589 | Fig. 2 "Our overall framework, including QA generation and evaluation."（注意：这是**数据构建+评测**流程，不是模型架构图） | https://arxiv.org/html/2505.12589v1/framework.png |
| `09-surveillancevqa-fig1-task-comparison.png` | 同上 | Fig. 1(b) "Normal and abnormal QA tasks." | https://arxiv.org/html/2505.12589v1/3task.png |

编号 `04` 有意留位：如需补 MM-LLMs 的 taxonomy 图（Fig. 3）或 Video-LLaVA Fig. 1（三种 LVLM 范式对比），可占用该编号。

## 与你那张图的对应关系

| 参考图 | 视频->Tokens | 独立 Query 输入 | LLM/推理体 | 输出 Answer | 领域 |
|---|---|---|---|---|---|
| `01` Video-ChatGPT Fig.1 | 有：帧->CLIP->Spatial/Temporal Pooling | 有：System Command + User Query | 有：Vicuna v1.1 | 有：Response | 通用 |
| `02` Video-LLaVA Fig.2(a) | 有：帧->LanguageBind->Share Projection | 有：文本嵌入层 | 有：Vicuna v1.5 | 有 | 通用 |
| `05` MM-LLMs Fig.2 | 有：视频走视觉编码器（均匀采样5帧） | 无（泛化描述） | 有：LLM Backbone | 有：文本输出 | 通用（形式化） |
| `07` NaviMind Fig.4 | 有：Camera->Visual Tokens | 有：User Query | 有：5 个 agent（Router/Captioner/Reasoner/Grader/Summary） | 有：Output | 水域/海事 |
| `08` SurveillanceVQA Fig.2 | 有：Video Sources | 无（数据构建视角） | 无 | 有：Answer | 监控视频 |

**结论**：`01` 或 `02` 用于「端到端 VLM 范式」对照；`07` 是与 SeaAgent 最直接的对标（同为水域视频问答、同为多智能体）；`08` 用于监控域的数据/评测对标。

## 自绘技术路线图（非引用件，可直接用于论文）

| 文件 | 内容 | 规格 |
|---|---|---|
| `10-seaagent-3subagent-architecture.png` | SeaAgent 三子智能体架构：Planning / Execution / Reflection | 5850x2430，300 dpi |
| `10-seaagent-3subagent-architecture.svg` | 同上矢量版，可进 LaTeX / Illustrator 继续改 | 1404x583 pt |
| `draw_seaagent.py` | 生成上述两张图的脚本（改字改色改布局后重跑即可） | 依赖 matplotlib |

**图面约定（与代码一致，避免评审追问）**

- `Planning Agent`：产出初始 plan，**不调用任何工具**（本图因此不接工具层）。
- `Execution Agent`：主力执行者；`TodoList middleware -> write_todos` 高亮框对应 `harness/middleware.py:66`，**plan 的实时状态由它维护**。
- `Reflection Agent`：验收判定何时可以退出（`skills/reflect_agent/`），负责证据聚合；不通过则回炉 `re-plan / re-execute`（有界轮次）。
- **证据富化走代码路径**：`show_evidence` 把关键帧、先验库参考图渲染到前端证据面板（`EvidenceWrapUpMiddleware` 收尾把关）。
- 原 `track_scout / registry_checker / visual_prover` 三个从智能体**收进 Execution Agent**，作为它的领域工具，不再单独成框。

重跑脚本：

```powershell
py reference-figures\draw_seaagent.py
```

## 引用条目

```bibtex
@article{maaz2023videochatgpt,
  title  = {Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models},
  author = {Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Khan, Fahad Shahbaz},
  journal= {arXiv preprint arXiv:2306.05424},
  year   = {2023}
}

@article{lin2023videollava,
  title  = {Video-LLaVA: Learning United Visual Representation by Alignment Before Projection},
  author = {Lin, Bin and Zhu, Bin and Ye, Yang and Ning, Munan and Jin, Peng and Yuan, Li},
  journal= {arXiv preprint arXiv:2311.10122},
  year   = {2023}
}

@article{zhang2024mmllms,
  title  = {MM-LLMs: Recent Advances in MultiModal Large Language Models},
  author = {Zhang, Duzhen and Yu, Yahan and Li, Chenxing and Dong, Jiahua and Su, Dan and Chu, Chenhui and Yu, Dong},
  journal= {arXiv preprint arXiv:2401.13601},
  year   = {2024}
}

@article{liu2025surveillancevqa,
  title  = {SurveillanceVQA-589K: A Benchmark for Comprehensive Surveillance Video-Language Understanding with Large Models},
  author = {Liu, Bo and Qiao, Pengfei and Ma, Minhan and Zhang, Xuange and Tang, Yinan and Xu, Peng and Liu, Kun and Yuan, Tongtong},
  journal= {arXiv preprint arXiv:2505.12589},
  year   = {2025}
}

@article{guan2026watervideoqa,
  title  = {WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents},
  author = {Guan, Runwei and Liang, Shaofeng and Ouyang, Ningwei and Fei, Weichen and Yao, Shanliang and Dai, Wei and Ge, Chenhao and Sun, Penglei and Zhu, Xiaohui and Huang, Tao and Liu, Ryan Wen and Xiong, Hui},
  journal= {arXiv preprint arXiv:2602.22923},
  year   = {2026}
}
```

## 复现方式

- 直链图：`Invoke-WebRequest` 直接抓取（URL 见上表）。
- Video-ChatGPT Fig.1：`pdfimages -f 4 -l 4 -png -j videocaptiongpt.pdf`，抽内嵌原始位图。
- Video-LLaVA Fig.2：该论文 arXiv HTML 图片直链已 404（`html/2311.10122v3/...`），改为 `pdftoppm -f 5 -l 5 -r 300 -png` 渲染第 5 页，再按 110dpi 坐标 `(92,112)-(505,360)` 等比裁剪。

