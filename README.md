# AI Agents/Agentic Workflow/大模型微调项目集合
# 1.多轮对话 LangGraph Agent智能客服
- 最新个人运行chatbot.ipynb
- chatbot.ipynb

Python/LangGraph/LangChain/langchain-openai/llangchaon-community/langchain-core/TavilySearchAPI/TypedDict/Annotated/PYthon.dispaly/os

**核心功能**:基础对话能力/集成外部工具/人工介入/历史对话查询

# 2.基于crewAI自动化写作应用实战

- 2.基于CrewAI自动化写作应用实战.ipynb
- 额度用完了，Azure配置有点问题，解决：https://github.com/gongfan1213/xiangmu/wiki/2.%E9%A1%B9%E7%9B%AE2AzureAPI.md

Python/CrewAI/Azure OpenAi/Agent/Task/Crew

**核心功能** ：自动化写作系统，需要两个Agent，故事内容构思者和故事内容协作者，两个Task:产出故事内容构思、根据构思编写故事，Crew组合团队并且执行、kickoff运行任务
# RAG

https://github.com/FareedKhan-dev/all-rag-techniques#
# 3.最简单的RAG的极简实现
- 3_simple_rag_副本 (1).ipynb（已经全部跑通的)
- 3_simple_rag.ipynb

# 4.基于百分位数法的语义分块
- 4_语义分块(已经跑通的).ipynb
- 4_semantic_chunking.ipynb

# 5.分块评估

评估响应的忠实度和相关性

- 5_chunk_size_selector_副本.ipynb
- 5_chunk_size_selector.ipynb

# 6_context_enriched_rag上下文增强检索调用
- 6_context_enriched_rag_副本 (1).ipynb
- 6_context_enriched_rag_副本.ipynb
# 7-CCH在简单rag中应用contextual_chunk_headers_rag
- 7_contextual_chunk_headers_rag_副本.ipynb
- 7_contextual_chunk_headers_rag_.ipynb

# 8_doc_augmentation_rag_问题增强RAG
- 8_doc_augmentation_rag_副本 (1).ipynb
- 8_doc_augmentation_rag.ipynb
# 9_query_transform三种查询转换方法|查询重写|回溯提示|子查询分解
- 9_query_transform_副本.ipynb
- 9_query_transform.ipynb
# 10_reranker_副本.ipynb基于LLM和keywords的重排序策略
- 10_reranker_副本.ipynb
- 10_reranker.ipynb
# 11_rse变种最大子数组
- 11_rse_副本.ipynb
- 11_rse.ipynb
# 12_contextual_compression三种上下文压缩summery/extract/selective
- 12_contextual_compression_副本.ipynb
- 12_contextual_compression.ipynb

# 13_feedback_loop_rag基于反馈的主动进化的RAG
- 13_feedback_loop_rag.ipynb
- 13_feedback_loop_rag副本.ipynb
# 14_adaptive_rag_增强型RAG系统的自适应检索
对查询类型进行分类（事实型、分析型、观点型或上下文型）,选择合适的检索策略,执行专门的检索技术,生成定制化的回答
- 14_adaptive_rag_副本 (1).ipynb
- 14_adaptive_rag.ipynb
# 15.Self-RAG自适应RAG

检索必要性判断、文档相关性评估、回答依据评估、回答实用性评分

-  15_self_rag.ipynb
-  15_self_rag_副本.ipynb

# 16-Langgraph-basiclanggraph基础

幻觉判断，人机交互，条件边

原先:
https://github.com/langchain-ai/langgraph-101?utm_source=chatgpt.com

- 16langgraph_basics.ipynb
- 16_langgraph_basics副本.ipynb

# 17-命名分块（超级慢32分钟都没完)proposition_chunking
不能用o1

用gpt4o- mini的结果:

https://github.com/gongfan1213/lianxi/tree/main/RAG/%E5%91%BD%E5%90%8D%E5%88%86%E5%9D%97

- 17_proposition_chunking_副本.ipynb
- 17_proposition_chunking.ipynb

# 18-多模态RAG_multimodel_rag

实现对图像的提取，豆包

Doubao-1.5-vision-pro-32k

- 18_multimodel_rag副本.ipynb
- 18_multimodel_rag.ipynb
# 19_fusion_rag融合检索
BM25关键词检索和向量检索结合

- 19_fusion_rag_副本.ipynb
- 19_fusion_rag.ipynb

# 20_graph_rag_图结构RAG
- 20_graph_rag_副本.ipynb
- 20_graph_rag.ipynb
# 21-分层RAG hierarch_rag
- 21_hierarchy_rag.ipynb
- 21_hierarchy_rag_副本.ipynb

# 22_HyDE_rag_假设理论RAG
- 22_HyDE_rag_副本.ipynb
- 22_HyDE_rag.ipynb
# 23_网络混合检索RAG
- 23_crag_副本.ipynb
- 23_crag.ipynb

# 24_rag_with_rl基于强化学习的rag
- 24_rag_with_rl.ipynb
- 24_rag_with_rl_副本.ipynb

# 25_function_call
- 训练一个工具调用模型
# 26_微调项目-Deepseed
- 使用 DeepSpeed 框架进行大规模模型训练的完整示例和配置
```
deepspeed/
├── README.md                    # 安装和配置指南
├── train_on_one_gpu.sh         # 单GPU训练脚本
├── train_on_multi_nodes.sh     # 多节点分布式训练脚本
├── config/                     # DeepSpeed配置文件
│   ├── ds_config_zero2.json    # ZeRO-2优化配置
│   └── ds_config_zero3.json    # ZeRO-3优化配置
└── translation/                # 翻译任务示例
    ├── run_translation.py      # 翻译训练脚本
    ├── requirements.txt        # 依赖包列表
    └── README.md              # 翻译任务说明
```
# 27-量化

- `bits_and_bytes.ipynb`：演示使用 bitsandbytes（bnb）进行 8bit/4bit 推理量化（不改权重，侧重加载/显存优化）。
- `AWQ-opt-125m.ipynb`：在小模型上跑通 AWQ 权重量化的最小可复现流程（W4A16 常见），便于理解全流程。
- `AWQ_opt-2.7b.ipynb`：在更大 OPT-2.7B 上执行 AWQ，覆盖校准、量化、保存、评测与推理。
- `AutoGPTQ_opt-2.7b.ipynb`：使用 AutoGPTQ 对 OPT-2.7B 做 GPTQ 权重量化的端到端流程（超详细，含多种配置与评测，体量最大）。
- `docs/images/qlora.png`：QLoRA 相关概念图（用于理解 LoRA+NF4 的训练型量化思路，对上面几份“推理/离线权重量化”起到对比说明）。

# 28- 微调llama

# 29 - 微调抽取式QA模型
- pipeline
- trasnformer
  
# 30-
  
# HKBU 可用模型列表
![image](https://github.com/user-attachments/assets/a33a1360-e981-40ce-93e0-2e79bf237bc7)

![image](https://github.com/user-attachments/assets/d0b8aa7c-e62e-484f-bd49-ba976383a214)

### Command line - Curl
Launch a terminal, command prompt, or powershell on your personal computer.


Input the command given below

```
curl https://genai.hkbu.edu.hk/general/rest/deployments/gpt-4-o-mini/chat/completions?api-version=2024-05-01-preview \
    -H "Content-Type: application/json" \
    -H "api-key: <YOUR_API_KEY>" \
    -d '{"messages":[{"role": "user", "content": "Hello!"}]}'
```

```
import requests

apiKey = "xxxxxx"
basicUrl = "https://genai.hkbu.edu.hk/general/rest"
modelName = "gpt-4-o-mini"
apiVersion = "2024-05-01-preview"

def submit(message):
    conversation = [{"role": "user", "content": message}]
    url = basicUrl + "/deployments/" + modelName + "/chat/completions/?api-version=" + apiVersion
    headers = { 'Content-Type': 'application/json', 'api-key': apiKey }
    payload = { 'messages': conversation }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return 'Error:', response

result = submit("Hello!")
print(result)
```

如果你在 OpenRouter 上寻找 **完全免费的、适合构建 Agent 应用的模型**，以下这些都是 Reddit 社区推荐的优质选项，适合支持工具调用、任务规划、对话管理等 agent 功能：

---

## 🚀 免费且适合 Agent 开发的 OpenRouter 模型推荐

### **1. DeepSeek‑R1 Zero（DeepSeek R1）**

* 是 OpenRouter 提供的**免费模型**，特别适合具备推理和任务支持功能的 Agent 使用（如 function‑call、工具调用）([apidog][1])。
* Reddit 用户提到：“Deepseek‑R1 ... good when using atomic agents”；也容易集成到 Agent 框架中([Reddit][2])。

### **2. Meta Llama 3.3 70B Instruct**

* 支持高达 **100,000 token 的上下文**，适合复杂、长对话任务；目前以 `:free` 标记提供([Reddit][3])。
* Reddit 用户称其为“you can use them in ST without paying”，适合构建大型对话型 agent([Reddit][3])。

### **3. Google Gemini 系列（Gemini Flash / Gemini 2.0 Flash Thinking Experimental）**

* OpenRouter 上提供的**Experimental 免费版本**支持较高吞吐量和大上下文，可以构建高响应速度的 agent([Reddit][3])。
* 免费额度有限，但适合测试与轻量代理开发。

### **4. Qwen‑2 7B Instruct（\:free）**

* 小体量但支持指令式对话，适合轻量 agent、multimodal 或任务型场景([GitHub][4], [apidog][1])。
* 配合工具（如图像输入）可用于边缘部署任务。

### **5. Mistral 系列 free 模型（如 Mistral Small / Devstral Small）**

* 官方 `select-openrouter-model` 脚本中列出了多个免费 Mistral 小模型（24B / 7B），适合技术性任务或快速推理 agent([SPR][5])。

---

## 🛠 适配 Agent 开发的方案推荐

### ✅ Agent 框架与模型适配

* 可以结合 **LiteLLM + OpenRouter** 构建 agent，支持灵活切换不同模型（DeepSeek，Llama，Gemini 等）([Medium][6])。
* 社区流行的 Agent 框架如 **AutoGen**、**AgentLite**、**ModelScope‑Agent** 等，均可直接将上述模型作为底层 LLM 接入，进行任务规划、工具调用、多 agent 协同等开发([arXiv][7])。

---

## 🧪 社区真实反馈

* Reddit 用户提到 Free 模型 DeepSeek‑R1 在 Agent 场景中效果良好：“Deepseek‑R1 ... good when using atomic agents”，并容易通过 OpenRouter API 调用([Reddit][2])。
* 对 Llama 3.3 70B 的体验评价：“free offering”、上下文支持极大，适合 RP 和复杂场景优化([Reddit][3])。
* 也有用户表示：Gemini Flash Free、DeepSeek Chat Free 是使用体验较为稳定的免费模型，适合初期 agent 测试与开发([Reddit][8], [Reddit][3])。

---

## 📋 总结推荐表

| 模型（OpenRouter ID）                      | 免费标志   | 特点                           | 适合场景                  |
| -------------------------------------- | ------ | ---------------------------- | --------------------- |
| DeepSeek‑R1 Zero (*:free*)             | ✅ 免费   | 推理能力强，适合 function-call agent | 多 agent、复杂推理、工具调用     |
| meta-llama/llama-3.3-70b-instruct      | ✅ Free | 超长上下文支持（100k tokens）         | 长对话、RP、任务规划 agent     |
| google/gemini‑2.0‑flash‑*Experimental* | ✅ Free | 高吞吐、大 context、响应快            | 快速响应 agent，探索 GPT 型场景 |
| qwen/qwen‑2‑7b‑instruct:free           | ✅ Free | 小型模组，轻量指令处理能力                | 边缘部署、多模态、小 agent      |
| mistralai/mistral-small‑*:free*        | ✅ Free | 技术优化模型，性能适中                  | 技术文档生成、轻推理 agent      |

---

## 📌 如何开始使用？

1. 注册 [OpenRouter](https://openrouter.ai)，生成 API key（无需绑定信用卡使用 `:free` 模型）([Reddit][9], [arXiv][7], [Reddit][3], [GitHub][10], [Reddit][11], [SPR][5])。


   ([Reddit][11], [arXiv][12], [Reddit][8])。

---

如果你需要具体示例代码，或希望推荐如何配置 agent 来实现具体任务（如 function-calling、工具集成等），我也可以进一步帮你整理！

[1]: https://apidog.com/blog/free-ai-models/?utm_source=chatgpt.com "Best Free AI Models You Can Use on OpenRouter"
[2]: https://www.reddit.com/r/LangChain/comments/1icqkvm?utm_source=chatgpt.com "Best open source models to build complex agents alternative to gpt4o/Claude:"
[3]: https://www.reddit.com/r/SillyTavernAI/comments/1ixxeh2?utm_source=chatgpt.com "PSA: You can use some 70B models like Llama 3.3 with >100000 token context for free on Openrouter"
[4]: https://github.com/OpenRouterTeam/ai-sdk-provider?utm_source=chatgpt.com "GitHub - OpenRouterTeam/ai-sdk-provider: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of AI models through the OpenRouter chat and completion APIs."
[5]: https://spr.com/free-local-ai-agents-with-openrouter-ollama-and-crewai/?utm_source=chatgpt.com "Creating Free, Local AI Agents with OpenRouter, Ollama, and CrewAI - SPR"
[6]: https://medium.com/%40KumarPradosh/power-up-your-ai-agents-unlocking-a-universe-of-models-with-litellm-and-openrouter-7c25262cbbe9?utm_source=chatgpt.com "Power Up Your AI Agents: Unlocking a Universe of Models with LiteLLM and OpenRouter | by Pradosh Kumar | Jun, 2025 | Medium"
[7]: https://arxiv.org/abs/2308.08155?utm_source=chatgpt.com "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"
[8]: https://www.reddit.com/r/Chub_AI/comments/1j08vfl?utm_source=chatgpt.com "What's the actual best free model, API and configuration/preset?"
[9]: https://www.reddit.com/r/Chub_AI/comments/1dmqvpa?utm_source=chatgpt.com "Any good presets for Openrouter? Specifically nous-capybara 7B:free"
[10]: https://github.com/wheattoast11/openrouter-deep-research-mcp?utm_source=chatgpt.com "GitHub - wheattoast11/openrouter-deep-research-mcp: MCP Server that orchestrates research with Claude and Perplexity/GPT/Gemini automatically"
[11]: https://www.reddit.com/r/SillyTavernAI/comments/1fzk3uj?utm_source=chatgpt.com "Is openrouter free to use ?"
[12]: https://arxiv.org/abs/2402.15538?utm_source=chatgpt.com "AgentLite: A Lightweight Library for Building and Advancing Task-Oriented LLM Agent System"

以下是HKBU GenAI Platform的可用模型列表及学生每月 token 额度整理：  


### **一、可用模型列表**  
#### 1. **Azure OpenAI 系列**  
| 模型名称          | Token 限制 | 知识截止时间   | 支持视觉 | 特性描述                                                                 |  
|-------------------|------------|----------------|----------|--------------------------------------------------------------------------|  
| GPT-4o            | 128K       | 2023年10月     | 否       | 快速且支持基于视觉的任务                                                 |  
| GPT-4o Mini       | 128K       | 2023年10月     | 是       | 高效处理小型任务                                                         |  
| o1（仅API）       | 128K       | 2023年10月     | 是       | 增强推理能力                                                             |  
| o1-Mini（仅API）  | 128K       | 2023年10月     | 是       | 小尺寸推理模型，响应更快                                                 |  

#### 2. **Anthropic Claude 3 系列**  
| 模型名称              | 知识截止时间   | 支持视觉 | 特性描述               |  
|-----------------------|----------------|----------|------------------------|  
| Claude 3.5 Sonnect    | 2023年8月      | 是       | 快速且功能多样         |  
| Claude 3 Haiku        | 2023年8月      | 是       | 快速处理小型任务       |  

#### 3. **Google Gemini 系列**  
| 模型名称              | 知识截止时间   | 支持视觉 | 特性描述               |  
|-----------------------|----------------|----------|------------------------|  
| Gemini 1.5 Pro        | 2023年11月     | 是       | 复杂任务表现优异       |  
| Gemini 1.5 Pro Flash  | 2023年11月     | 是       | 快速且功能多样         |  

#### 4. **Facebook Llama 系列**  
| 模型名称              | 知识截止时间   | 支持视觉 | 特性描述               |  
|-----------------------|----------------|----------|------------------------|  
| Llama 3.1 405B        | 2023年12月     | 否       | 多语言对话优化         |  

#### 5. **DeepSeek 系列**  
| 模型名称              | 知识截止时间   | 支持视觉 | 特性描述               |  
|-----------------------|----------------|----------|------------------------|  
| DeepSeek-R1           | 2025年1月      | 否       | 中文能力突出，推理强   |  
| DeepSeek-V3           | 2024年7月      | 否       | 混合专家模型，中文强   |  

#### 6. **Qwen 系列**  
| 模型名称              | 知识截止时间   | 支持视觉 | 特性描述               |  
|-----------------------|----------------|----------|------------------------|  
| Qwen2.5-Max           | 2024年12月     | 否       | 高性能，复杂推理，中文理解强 |  
| Qwen-Plus             | 2024年12月     | 否       | 性能均衡，性价比高，中文支持强 |  


### **二、学生每月 token 额度（每个模型）**  
#### 1. **Azure OpenAI**  
| 模型名称          | 每月 token 额度 |  
|-------------------|----------------|  
| GPT-4o            | 500,000        |  
| GPT-4o Mini       | 8,000,000      |  
| o1                | 100,000        |  
| o1-Mini           | 500,000        |  

#### 2. **Anthropic**  
| 模型名称              | 每月 token 额度 |  
|-----------------------|----------------|  
| Claude 3.5 Sonnet     | 200,000        |  
| Claude 3 Haiku       | 2,000,000      |  

#### 3. **Google Gemini**  
| 模型名称              | 每月 token 额度 |  
|-----------------------|----------------|  
| Gemini 1.5 Pro        | 500,000        |  
| Gemini 1.5 Pro Flash  | 5,000,000      |  

#### 4. **Facebook Llama**  
| 模型名称              | 每月 token 额度 |  
|-----------------------|----------------|  
| Llama 3.1 405B        | 200,000        |  

#### 5. **DeepSeek**  
| 模型名称              | 每月 token 额度 |  
|-----------------------|----------------|  
| DeepSeek-R1           | 5,000,000      |  
| DeepSeek-V3           | 5,000,000      |  

#### 6. **Qwen**  
| 模型名称              | 每月 token 额度 |  
|-----------------------|----------------|  
| Qwen2.5-Max           | 1,500,000      |  
| Qwen-Plus             | 8,000,000      |  


### **三、其他注意事项**  
1. **额度共用规则**：API 服务与平台网页端共享每月 token 配额。  
2. **优化建议**：发送新消息前清空对话历史，可减少 token 消耗。  
3. **视觉功能**：部分模型（如 GPT-4o、Gemini 等）原生支持视觉理解，其他模型通过 OCR 处理文件（支持格式：图片、Office、PDF、txt 等）。  
4. **技术支持**：若有额度或 API 相关问题，可联系 ITO 服务中心（电话：3411 7899，邮箱：hotline@hkbu.edu.hk）。

# 可用模型一览表
以下是按模型类型分类整理的可用模型列表（包含原始所有模型，无遗漏）：


### **1. Claude系列（Anthropic）**
- `claude-3-5-sonnet-20240620`
- `claude-3-5-sonnet-20241022`
- `claude-3-7-sonnet-20250219`
- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`


### **2. DeepSeek系列**
- `deepseek-r1`
- `DeepSeek-R1-Distill-Qwen-32B`
- `DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-v3`
- `Pro-DeepSeek-R1`
- `Pro-DeepSeek-V3`


### **3. Doubao系列（豆包模型）**
- `Doubao-1.5-vision-pro-32k`（支持多模态）
- `Doubao-embedding`（嵌入向量模型）
- `Doubao-lite-128k`（长文本处理）
- `Doubao-lite-32k`
- `Doubao-lite-4k`
- `Doubao-pro-128k`（高性能版）
- `Doubao-pro-32k`
- `Doubao-pro-4k`


### **4. Gemini系列（Google）**
- `gemini-2.0-flash`
- `gemini-2.0-flash-lite`
- `gemini-2.5-flash-preview-04-17`（预览版）
- `gemini-2.5-pro-preview-03-25`（预览版）
- `gemini-2.5-pro-preview-05-06`（预览版）


### **5. GPT系列（OpenAI）**
- `gpt-3.5-turbo`（最常用的通用模型）
- `gpt-4`（高性能版）
- `gpt-4-32k`（超长上下文）
- `gpt-4-turbo`（优化版）
- `gpt-4.1`
- `gpt-4.1-mini`（轻量版）
- `gpt-4.1-nano`（超轻量版）
- `gpt-4o`（优化版）
- `gpt-4o-mini`（轻量优化版）


### **6. Kimi系列（字节跳动）**
- `kimi-latest`
- `kimi-thinking-preview`（预览版）


### **7. Moonshot系列（支持多模态）**
- `moonshot-small`
- `moonshot-v1-128k`（超长上下文）
- `moonshot-v1-128k-vision-preview`（多模态预览版）
- `moonshot-v1-32k`
- `moonshot-v1-32k-vision-preview`（多模态预览版）
- `moonshot-v1-8k`
- `moonshot-v1-8k-vision-preview`（多模态预览版）
- `moonshot-v1-auto`（自动优化版）


### **8. O系列**
- `o1`
- `o1-mini`（轻量版）
- `o3`
- `o3-mini`（轻量版）


### **9. Qwen系列（通义千问）**
- `qwen-max`
- `qwen-plus`
- `qwen-turbo`
- `qwen2.5-72b-instruct`（72B大模型）
- `qwen3-14b`
- `qwen3-235b-a22b`（235B超大模型）
- `qwen3-30b-a3b`
- `qwen3-32b`
- `qwen3-8b`


### **10. 嵌入向量模型（用于RAG检索）**
- `text-embedding-3-large`（大尺寸嵌入）
- `text-embedding-3-small`（小尺寸嵌入）
- `text-embedding-ada-002`（OpenAI经典嵌入模型）


### **注意事项**
1. **预览版模型**：如`gemini-2.5-pro-preview`、`moonshot-v1-128k-vision-preview`等，可能存在功能限制或不稳定。
2. **上下文长度**：部分模型（如`gpt-4-32k`、`Doubao-lite-128k`）支持超长上下文，适合处理长文本。
3. **多模态支持**：如`Doubao-1.5-vision-pro-32k`、`moonshot-v1-128k-vision-preview`可处理图像等非文本内容。


如果需要针对特定场景（如RAG检索、长文本处理、多模态）的模型推荐，可以进一步说明需求！

以下是对各系列模型的适用场景及原因的详细分析：


### **1. Claude系列（Anthropic）**
- **适用场景**：企业级长文本处理、合规文档分析、复杂逻辑推理  
- **原因**：
  - 具备超长上下文处理能力（如`claude-3-7-sonnet`支持数万token），适合法律合同、技术手册等长文档理解。
  - 强调“对齐性”设计，输出内容更符合人类价值观，适合金融、医疗等对合规性要求高的场景。
  - `claude-opus`系列强化了代码生成和数学推理能力，可用于技术开发辅助。


### **2. DeepSeek系列**
- **适用场景**：代码生成、模型蒸馏优化、垂直领域精调  
- **原因**：
  - `deepseek-r1`和`Pro-DeepSeek-R1`针对代码理解与生成优化，支持多语言编程任务。
  - 蒸馏版本（如`Distill-Qwen`）通过轻量化设计降低部署成本，适合边缘设备或实时推理。
  - `deepseek-v3`在通用对话和知识问答中表现均衡，适合企业级智能助手开发。


### **3. Doubao系列（豆包模型）**
- **适用场景**：多模态交互、长文本分析、嵌入式应用  
- **原因**：
  - `Doubao-1.5-vision-pro-32k`支持图像+文本多模态理解，适合电商商品描述生成、图文问答。
  - 长文本版本（如`Doubao-lite-128k`）支持128k token上下文，可处理学术论文综述、书籍摘要。
  - `Doubao-embedding`专为RAG检索优化，向量表征精度高，适合企业知识库构建。
  - 轻量化版本（如`Doubao-lite-4k`）适合移动端APP或低算力设备部署。


### **4. Gemini系列（Google）**
- **适用场景**：多模态创作、实时交互、科研辅助  
- **原因**：
  - `gemini-2.0`系列在图像理解、视频分析上表现突出，可用于广告创意生成、视频字幕总结。
  - 轻量化版本（如`flash-lite`）适合手机端实时对话，响应速度快。
  - 预览版（如`2.5-pro`）强化了科学计算和逻辑推理，适合科研数据处理、公式推导。


### **5. GPT系列（OpenAI）**
- **适用场景**：通用对话、内容创作、专业领域深度应用  
- **原因**：
  - `gpt-3.5-turbo`性价比高，适合聊天机器人、文案生成等通用场景。
  - `gpt-4`系列（如`gpt-4-32k`）支持超长上下文和复杂推理，适合法律文书分析、多轮专业咨询。
  - 轻量版（如`gpt-4.1-mini`）适合中小企业低成本部署，满足基础问答需求。
  - `gpt-4o`优化了工具调用能力，可与外部API结合实现数据查询、实时计算。


### **6. Kimi系列（字节跳动）**
- **适用场景**：中文场景交互、多轮思考推理  
- **原因**：
  - `kimi-latest`针对中文语义优化，对话流畅度高，适合社交媒体客服、内容审核。
  - `thinking-preview`支持分步思考（类似Chain of Thought），适合数学题解答、逻辑谜题推理。


### **7. Moonshot系列（多模态）**
- **适用场景**：多模态营销、长视频分析、实时视觉问答  
- **原因**：
  - 多模态版本（如`128k-vision-preview`）支持图像+长文本处理，可用于电商产品图自动描述、短视频脚本生成。
  - 超长上下文（`128k`）适合处理数小时的视频字幕转录与总结。
  - `auto`版本支持自动优化参数，降低企业调优门槛。


### **8. O系列**
- **适用场景**：轻量级应用、快速原型开发  
- **原因**：
  - 轻量版（如`o1-mini`）模型体积小，部署成本低，适合初创公司搭建基础问答系统。
  - 通用能力均衡，适合对性能要求不高的简单对话场景（如内部工具助手）。


### **9. Qwen系列（通义千问）**
- **适用场景**：中文企业服务、大模型精调、复杂任务处理  
- **原因**：
  - 超大模型（如`qwen3-235b`）具备超强知识存储和推理能力，适合金融风险预测、科研文献综述。
  - `qwen-turbo`优化了响应速度，适合实时客服、直播互动问答。
  - `qwen2.5-72b`等中大型模型支持深度领域精调（如医疗、工业），适配企业垂直需求。


### **10. 嵌入向量模型（RAG检索）**
- **适用场景**：企业知识库检索、语义相似度匹配、文档聚类  
- **原因**：
  - `text-embedding-3-large`等模型将文本转化为高维向量，支持精准的语义检索（如法律案例匹配、客服知识库查询）。
  - `ada-002`作为经典模型，兼容性强，适合与各类LLM结合构建RAG系统。


### **场景化选择建议**
- **长文本处理**：优先选择`claude-3-7`、`Doubao-lite-128k`、`gpt-4-32k`，上下文长度决定处理效率。
- **多模态应用**：`Doubao-1.5-vision`、`Gemini-2.0`、`Moonshot-vision`支持图像/视频理解，适合创意类场景。
- **企业级RAG**：`Doubao-embedding`+`Qwen3`或`text-embedding-3-large`+`GPT-4`组合，兼顾向量检索精度与回答质量。
- **轻量化部署**：`DeepSeek-R1-Distill`、`gpt-4.1-mini`、`O1-mini`适合算力有限的中小团队。

