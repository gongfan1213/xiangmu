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

