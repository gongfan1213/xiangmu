### 目录总览（`/Users/gongfan/projects/LLM-quickstart/transformers`）
- `fine-tune-QA.ipynb`: 抽取式问答（SQuAD）微调全流程。涵盖长文档切片（`max_length/doc_stride`）、`offset_mapping` 对齐、`Trainer` 训练、logits 后处理重组答案、SQuAD 指标评估（EM/F1）。适合已掌握基础的同学深入实践。
- `fine-tune-quickstart.ipynb`: 微调快速上手（文本分类场景）。包含数据集加载、`AutoModelForSequenceClassification`、`Trainer` 配置、评估指标。对初学者更友好。
- `pipelines.ipynb`: Transformers `pipeline` 基础用法速览。涵盖常见任务（情感分析、生成、问答、NER、翻译、摘要等）的“一行推理”。
- `pipelines_advanced.ipynb`: `pipeline` 进阶。包含批处理、设备放置、手动模型/分词器、返回结构自定义、流式/高阶控制等实战技巧。
- `docs/`: 笔记本内嵌示意图与流程图资源（如 `images/question_answering.png` 等）。
- `data/`: 演示或测试用的小型样本（如音频/图片），供部分 notebook 读取。

### 推荐使用顺序
1) `pipelines.ipynb` → 2) `pipelines_advanced.ipynb` → 3) `fine-tune-quickstart.ipynb` → 4) `fine-tune-QA.ipynb`

### 在 conda 虚拟环境运行的要点
- 保持你当前的 conda 环境已激活；显存有限时优先调小 `per_device_train_batch_size` 与 `max_length`，或开启混合精度（`fp16=True`/`bf16=True`）。
- 若 `datasets.map` 结果没更新，给 `map(..., load_from_cache_file=False)` 强制重算。
- 确认使用 Fast 分词器（`PreTrainedTokenizerFast`），否则 `offset_mapping` 不可用（影响问答后处理）。
- SQuAD QA 微调关键：`return_overflowing_tokens=True` + `doc_stride` 切片；验证集特征要把非上下文 `offset_mapping` 置 `None`，后处理只在上下文中取答案。

### 常见坑位
- `CUDA OOM`: 降低 `batch_size`/`max_length`，或用梯度累积与混合精度。
- SQuAD v2 开关：`squad_v2=True` 时需比较空答案分数（CLS 索引）以决定是否输出空字符串。
- 断点续训：`output_dir` 下已有 `checkpoint-*` 时，指定 `resume_from_checkpoint` 更稳妥。

需要的话，我可以为 `transformers/` 目录补一份简洁的 `README.md`（含各 notebook 目的、运行前置与常见参数）或把 `fine-tune-QA.ipynb` 的评估后处理封装成可复用函数，便于你在其它模型上快速复用。
