## 目录概览（quantization/）

- `bits_and_bytes.ipynb`：演示使用 bitsandbytes（bnb）进行 8bit/4bit 推理量化（不改权重，侧重加载/显存优化）。
- `AWQ-opt-125m.ipynb`：在小模型上跑通 AWQ 权重量化的最小可复现流程（W4A16 常见），便于理解全流程。
- `AWQ_opt-2.7b.ipynb`：在更大 OPT-2.7B 上执行 AWQ，覆盖校准、量化、保存、评测与推理。
- `AutoGPTQ_opt-2.7b.ipynb`：使用 AutoGPTQ 对 OPT-2.7B 做 GPTQ 权重量化的端到端流程（超详细，含多种配置与评测，体量最大）。
- `docs/images/qlora.png`：QLoRA 相关概念图（用于理解 LoRA+NF4 的训练型量化思路，对上面几份“推理/离线权重量化”起到对比说明）。

---

## 前置知识与选型

- **推理量化 vs 训练量化**
  - 推理量化（本目录主线）：将已训好的全精度权重压缩到更低比特（如 4bit/8bit）用于部署推理。典型：AWQ、GPTQ、bnb 量化加载。
  - 训练量化（如 QLoRA）：在保持模型推理能力的同时，用低比特权重和优化器状态进行微调训练。本目录未直接微调，但图示在 `docs/images/qlora.png`。
- **三种方法定位**
  - bitsandbytes（bnb）量化加载：最容易上手，适合快速节省显存做推理与微调（QLoRA）。不改权重文件。
  - AWQ：Activation-aware Weight Quantization，离线权重量化，需要校准集；对多种解码器效果稳定，兼顾保真与速度。
  - GPTQ：基于二阶近似的权重量化方案，AutoGPTQ 提供成熟易用的实现，广泛用于 LLaMA/OPT 等系列。

---

## 环境与依赖（通用）

- 建议准备：
  - Python 3.10+，CUDA 对应匹配的 PyTorch（≥2.0 常见）。
  - `transformers`、`accelerate`、`datasets`、`evaluate`、`bitsandbytes`、`auto-gptq`、`awq`（或其实现库）。
- 典型兼容注意：
  - `bitsandbytes` 需与 CUDA 匹配；Mac M系列不支持 GPU bnb，仅 CPU/Fake-quant 备用。
  - `auto-gptq` 的 Triton/CUDA 依赖需与环境严格匹配；不同版本的 `torch`、`triton` 有适配要求。
  - AWQ 的 kernel 或者量化算子依赖，注意对应库版本与 GPU 架构支持（例如 sm_80、sm_86）。

---

## 共用准备步骤（强烈建议）

1. 准备/选择模型（如 `facebook/opt-125m`、`facebook/opt-2.7b` 或其他 Hugging Face 上的 OPT/LLaMA/GLM 等）。
2. 准备一小段“校准数据”（几十～几百条文本）：
   - AWQ/GPTQ 都会用到，用来拟合/感知激活分布以决定更合理的量化尺度。
   - 可用 `langchain/tests/*.txt` 这种文本或你自己的业务数据。
3. 基线评测（可选但推荐）：
   - 在量化前做一次 perplexity/零样本任务/简单问答，以便对比量化后效果。
4. 显存与性能基线记录：
   - 记录未量化时的峰值显存与吞吐（tokens/s），后续对比量化收益。

---

## `bits_and_bytes.ipynb`：8bit/4bit 推理量化（加载时量化）

- 目标：不修改权重文件，通过 bnb 配置把模型以 8bit 或 4bit 的形式加载到 GPU，显著降低显存占用。
- 关键点：
  - `from transformers import BitsAndBytesConfig`
  - 常见配置：
    - 8bit：`load_in_8bit=True`
    - 4bit：`load_in_4bit=True` 并指定 `bnb_4bit_compute_dtype=torch.bfloat16` 或 `torch.float16`
    - 可选量化类型：`bnb_4bit_quant_type="nf4"`（QLoRA 常用）或 `"fp4"`
  - 典型加载：
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-2.7b",
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16
    )
    ```
- 适用场景：
  - 快速做推理/少量微调，免去了离线量化流程。
  - 显存节省大，但有时速度略有损耗，且不同 GPU/驱动下稳定性有差异。
- 注意事项：
  - `bitsandbytes` 与 `CUDA` 版本强耦合；若加载失败，先检查 `nvidia-smi`、`torch.cuda.is_available()` 与 bnb 版本匹配。
  - 4bit 下优先用 `bfloat16` 计算精度。

---

## `AWQ-opt-125m.ipynb`：AWQ 最小流程（上手）

- 目标：用小模型（OPT-125M）跑通 AWQ 全流程，理解每一步的作用。
- 核心流程：
  1. 加载 FP16 模型与分词器。
  2. 准备一小段校准文本，构造 `DataLoader` 走一遍前向，统计/感知激活。
  3. 调用 AWQ 的量化 API：对权重执行分组量化（常用 W4A16，即权重 4bit、激活 16bit）。
  4. 保存量化后的权重（常见为 `safetensors` 分片）。
  5. 评测：困惑度（PPL）对比、简单推理对比。
- 产出：
  - 一个可直接加载的“AWQ 量化版”模型目录（体积显著缩小）。
- 优势：
  - 对解码器类模型稳定；能保持相对较好精度的同时降低显存/存储。
- 注意：
  - 校准数据质量与数量会影响效果；建议覆盖典型语料风格。

---

## `AWQ_opt-2.7b.ipynb`：在更大模型上做 AWQ

- 与 125M 一致，但更贴近真实部署规模，重点在于：
  - 数据管线优化（校准集更接近业务分布）。
  - 显存与吞吐记录（对比 FP16、AWQ-W4）。
  - 量化参数（如每组大小 group size、对称/非对称、逐层/逐通道策略）对精度与速度的取舍。
- 典型推理加载（量化后）：
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model_dir = "path/to/awq-quantized-opt-2.7b"
  tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
  model = AutoModelForCausalLM.from_pretrained(
      model_dir,
      device_map="auto",
      torch_dtype=torch.float16  # 推理时激活仍常用 fp16/bf16
  )
  ```
- 常见问题：
  - 特定层（如 embedding/output head）是否量化、是否跳过，可能影响质量；Notebook 中通常提供可配置开关。
  - 某些自定义 kernel 需要特定 GPU 架构；如遇到异常，先降级到 PyTorch 算子路径验证正确性。

---

## `AutoGPTQ_opt-2.7b.ipynb`：AutoGPTQ 端到端 GPTQ 量化（超详解）

- 内容最全，通常包括：
  1. 加载 FP16 模型与分词器。
  2. 准备校准数据集（`datasets`/本地文本）。
  3. 设置 GPTQ 配置：
     - 量化比特：通常 4bit
     - 分组大小：例如 128/64
     - 是否对称、是否 per-channel、是否 damp（阻尼）等
     - 是否启用 Triton/EXLlama kernel 加速（取决于 GPU/环境）
  4. 执行 GPTQ 量化，得到量化权重。
  5. 保存权重到目录（`safetensors`）。
  6. 评测：PPL、推理速度、显存占用对比。
  7. 推理：用 `AutoGPTQForCausalLM.from_quantized` 直接加载量化模型。
- 典型加载（推理）：
  ```python
  from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
  from transformers import AutoTokenizer

  model_dir = "path/to/gptq-quantized-opt-2.7b"
  tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
  model = AutoGPTQForCausalLM.from_quantized(
      model_dir,
      device_map="auto",
      torch_dtype="auto",
      use_cuda_fp16=True,   # 或 bf16，根据你的环境
      inject_fused_attention=True,   # 视 kernel 支持
  )
  ```
- 选择与调优要点：
  - 组大小越小越细粒度，精度更好但速度略慢；反之亦然。
  - `use_triton`、`inject_fused_attention` 等开关会影响速度与稳定性，需结合 GPU 架构测试。
- 常见报错与处理：
  - Triton 编译失败/不可用：固定 `torch` 与 `triton` 版本组合；必要时关闭 triton 相关加速开关先跑通。
  - 加载时 Key/shape 不匹配：确认量化时使用的模型版本、分支与推理时一致。

---

## AWQ vs GPTQ vs bnb（何时选用）

- **权重量化离线产物（AWQ/GPTQ）**
  - 优点：部署时不依赖 bnb，加载即低显存；可与高性能 kernel 深度结合；部署时更“干净”。
  - 适用：需要产出“可交付”的低比特权重，复用在多台机器或服务。
- **加载时量化（bnb）**
  - 优点：简单、快速、无需离线处理；最适合实验/调研/小规模微调。
  - 适用：显存紧张但需要快速迭代；或做 QLoRA（训练型量化）。

---

## 从 0 到 1 的实操顺序（建议）

1. 先用 `bits_and_bytes.ipynb` 跑一个 4bit 推理加载，确认环境与 GPU 可用，记录显存与吞吐。
2. 准备一份你业务相关的 200～1000 条文本作为校准集。
3. 选一个模型规模尝试 AWQ：
   - 小规模先用 `AWQ-opt-125m.ipynb`，跑通后切换到 `AWQ_opt-2.7b.ipynb`。
   - 保存量化模型并对比：PPL、任务指标、显存、吞吐。
4. 尝试 AutoGPTQ：
   - 跑 `AutoGPTQ_opt-2.7b.ipynb`，调节组大小、对称性、Triton 开关，找到最优平衡点。
5. 横向对比：
   - 以相同评测脚本对比 FP16、bnb-4bit、AWQ-4bit、GPTQ-4bit 的质量/速度/显存。
6. 选型落地：
   - 若要产出交付权重：选择 AWQ 或 GPTQ。
   - 若要快速场景验证或 QLoRA 微调：选择 bnb 4bit 加载。

---

## 常见坑与优化建议

- **CUDA/驱动/torch 版本不匹配**：部署前固定 “CUDA + nvidia driver + torch + triton + bitsandbytes/auto-gptq” 的版本矩阵。
- **bf16 vs fp16**：4bit 下通常 `bf16` 稳定性与精度更好（A100/H100/RTX40 系列支持更佳）。
- **校准数据**：风格与下游任务越接近越好；数量不必太大但要覆盖多样性。
- **层跳过策略**：embedding/output 层常被跳过量化；可通过实验验证对精度影响。
- **加载加速**：AutoGPTQ 的 fused attention、Triton、ExLlama 等可提速，但需谨慎验证稳定性。
- **显存对齐**：记录“最大 batch、最大 context length”下的显存峰值，以便合理设置服务参数。

---

## `docs/images/qlora.png` 的意义

- 图示 QLoRA 的核心思想：低比特权重（如 NF4）+ LoRA Adapter 在 bf16/fp16 计算下进行训练，显著降低微调成本。
- 与本目录的“推理/离线权重量化”互补：前者偏训练方案，后者偏部署压缩。两者可串联：先用 QLoRA 微调得到 LoRA 适配器，再做合并与（或）离线权重量化。

---

## 你可以立即尝试的最小示例（推理）

- bnb 4bit（最快上手）：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "facebook/opt-2.7b"
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant_cfg)

inputs = tok("Hello, how are you?", return_tensors="pt").to(model.device)
out_ids = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out_ids[0], skip_special_tokens=True))
```

---

## 结论与建议

- 想“最快”省显存跑推理：先用 `bits_and_bytes.ipynb` 的 4bit 加载。
- 想“交付可部署的量化权重”：优先尝试 `AWQ_opt-2.7b.ipynb` 与 `AutoGPTQ_opt-2.7b.ipynb`，分别对比精度与速度，按业务指标定夺。
- 持续迭代：用统一评测脚本记录 PPL/准确率/时延/吞吐/显存，形成你的量化基准线，逐步优化配置参数。

如果你告诉我目标模型与 GPU 环境（显卡型号、显存大小、CUDA/torch 版本），我可以给你定制化的具体参数与执行顺序。
