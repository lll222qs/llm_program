
# 🚀 Industrial Vision QA & Maintenance System based on Agentic RAG
### (基于 Agentic RAG 的工业视觉质检与智能运维系统)



## 📌 项目概述

本项目旨在模拟高端制造业（如汽车、半导体）中智能运维专家的日常工作流。当生产线出现异常（如钢材表面出现划痕、夹杂等缺陷）时，系统能够像一个经验丰富的工程师一样：**看图 -> 查手册 -> 思考 -> 给出专业诊断与处理建议**。

不同于简单的“看图说话”Demo，本项目是一个具备 **“视觉感知 + 知识推理 + 自动闭环”** 能力的完整系统。

---

## 🌟 核心亮点

1. **Agentic RAG 核心闭环**：
   Agent 不仅能检索本地知识库（如维修手册），还能主动调用计算机视觉（CV）模型进行二次确认，实现了真正的“图文互证”。
2. **多模态工业场景输入**：
   支持文本指令与工业缺陷图片（Base64 / 本地路径）的无缝结合，模拟真实工单的输入形式。
3. **MLOps 全流程工程化**：
   从模型训练、版本管理到使用 Docker 进行自动化容器化部署，展示了完整的工程化落地能力。
4. **工业级防呆与异常处理**：
   针对大模型常见的幻觉问题，设计了多重代码级防呆机制，确保在生产环境中运行的稳定性。

---

## 🏗️ 系统架构设计 (三层架构)

本项目采用清晰的三层架构设计，融合了前沿的 AI 技术与成熟的工程实践：

| 层级 | 核心技术栈 | 模块作用 |
| :--- | :--- | :--- |
| **工程层 (Ops)** | FastAPI, Docker, Qdrant, MLflow | 提供 RESTful 接口，模型版本管理，向量数据库与自动化部署。 |
| **认知层 (Brain)** | LangGraph, Qwen2.5-7B, ChromaDB | Agent 的大脑，负责理解上下文、规划路径、检索知识库及逻辑推理。 |
| **感知层 (Vision)** | YOLOv8 (ONNX Runtime), OpenCV | 毫秒级识别工业缺陷（如夹杂、裂纹），提取结构化的坐标与置信度数据。 |

---

## 🛠️ 技术栈

* **大模型框架**：LangGraph, LangChain, Ollama (Qwen2.5)
* **视觉模型**：YOLOv8, ONNX Runtime, OpenCV, Numpy
* **知识库/向量检索**：Qdrant, ChromaDB
* **Web 框架**：FastAPI, Uvicorn, Pydantic
* **部署与工程**：Docker, Docker-Compose

---

## 🚀 快速上手 (Quick Start)

本项目已完全容器化，使用 Docker Compose 一键启动，无需配置复杂的本地 Python 环境。

### 1. 环境准备

* 安装 [Docker](https://www.docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/)
* 准备一张工业钢材缺陷图片（或测试图片）

### 2. 启动项目

在项目根目录下执行以下命令，系统将自动拉起 FastAPI 服务、Qdrant 向量数据库：

```bash
docker-compose up --build -d
```

### 3. 访问 API 文档

服务启动后，在浏览器中打开：
`http://localhost:8000/docs`

你可以在交互式文档中直接调用 `/detect_and_advice` 接口，上传图片进行测试。

---

## 🔥 核心难点与工程化实践 (Project Highlights)

在开发过程中，我们遇到了大模型落地过程中最典型的“状态失忆”与“工具调用幻觉”问题，并通过以下手段解决：

### 🧠 LangGraph 状态机的深度定制
* **问题**：默认的 State 在处理多轮对话时，会导致历史记录被截断，模型“忘记”之前的检测结果。
* **解决**：自定义了 `AgentState`，使用 `Annotated[Sequence, add]` 累加器确保历史消息链的完整；将真实图片路径（`image_path`）作为 State 的持久化字段，解决了复杂异步框架中上下文变量丢失的难题。

### 🛡️ 工业级工具防呆机制 (Anti-Hallucination)
* **问题**：大模型在获取检测结果后，容易因为逻辑推理惯性，再次尝试调用视觉工具，并传入无效的占位符（如 `img-0` 或 `/path/to/image.jpg`）。
* **解决**：在 `detect_defect` 工具内部实现了严格的“代码级防呆”逻辑。通过 State 校验与真实路径回退机制，当检测到无效输入时直接拦截并返回错误，杜绝了无意义的重复 ONNX 推理，显著降低了系统算力开销。

### 🤖 动态工具分发节点 (Dynamic Tool Node)
* **创新点**：摒弃了硬编码的工具执行逻辑。
* **解决**：在 `tools_node` 中根据 `tool_calls` 的名称进行动态路由，使得 Agent 能够根据任务需求，灵活地在“视觉检测”、“知识检索”、“异常兜底”等工具间自由切换。

---

## 📁 目录结构

```
├── agent_day2_graph.py       # LangGraph 核心状态机与 Agent 决策逻辑
├── agent_day2_vision.py      # 视觉感知模块（ONNX 推理与预处理）
├── rag_knowledge.py          # RAG 知识库检索逻辑
├── main.py                   # FastAPI 主入口与业务路由
├── requirements.txt          # Python 依赖清单
├── Dockerfile                # 容器化构建文件
├── docker-compose.yml        # 编排文件
└── README.md                 # 项目说明文档
```

---

## 🤝 贡献与反馈

欢迎在 Issues 中提交问题，或者通过 Pull Requests 贡献你的代码。让我们一起探索 Agentic RAG 在工业领域的无限可能！

---
