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



