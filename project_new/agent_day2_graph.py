
# agent_day2_graph.py  (修复版)
# LangGraph Agent 核心逻辑 - Agentic RAG + 内存 Checkpointer
#
# 本次修复的两个致命问题：
#   [FIX-1] AgentState.messages 的 Annotated 第二个参数原本写的是字符串 "对话历史"，
#           那不是 reducer，导致 messages 通道退化成"覆盖写"：
#           agent 返回一次 -> 历史被替换成 1 条 AI 消息
#           tools  返回一次 -> 历史又被替换成 1 条 ToolMessage
#           于是模型每一跳都"失忆"，看不到用户原图请求、也看不到自己刚拿到的检测结果，
#           只能重复调用 detect_defect 并瞎编 /path/to/image.jpg 这种占位路径。
#           ==> 必须改成 Annotated[..., add]（你 import 了 add 但一直没用上）。
#   [FIX-2] detect_defect 的工具参数不能交给大模型填。改为：工具签名保持 image_path: str，
#           由 tools_node 在 invoke 之前用 state["image_path"] 强制覆盖 args，
#           大模型无论传什么占位符都会被真实落盘路径顶掉；再用 detect_result 缓存
#           彻底杜绝第二次重复推理。
#   [FIX-3] detect_defect 请保持 agent_day2_vision.py 里 image_path: str 的签名
#           （不要改成 state: dict，@tool 会把 schema 变成要求 LLM 传 state 对象，必然对不上）。

import os
import json
from typing import Annotated, Optional, Sequence, TypedDict, Literal
from operator import add   # <<< FIX-1 的关键：add 才是 reducer

# ================= 1. 依赖导入与防御性检查 =================
try:
    from langchain_ollama import ChatOllama
    USE_NATIVE_OLLAMA = True
except ImportError:
    try:
        from langchain_openai import ChatOpenAI as ChatOllama  # Fallback
        USE_NATIVE_OLLAMA = False
    except ImportError:
        raise ImportError("请安装 langchain-ollama 或 langchain-openai")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# 尝试导入业务模块
try:
    from agent_day2_vision import detect_defect as vision_detect, _current_image_path
    VISION_TOOL_AVAILABLE = True
except ImportError:
    VISION_TOOL_AVAILABLE = False
    _current_image_path = None          # <<< 兜底，避免后面引用未定义变量
    print("[WARN] 未找到 agent_day2_vision，将使用内嵌 YOLO 兜底")

try:
    from rag_knowledge import search_manual as rag_search
    RAG_TOOL_AVAILABLE = True
except ImportError:
    RAG_TOOL_AVAILABLE = False
    print("[WARN] 未找到 rag_knowledge，RAG 功能不可用")

# ================= 2. 环境变量配置 =================
from dotenv import load_dotenv
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ================= 3. 工具定义 (动态构建) =================
tools = []

# 3.1 注册视觉工具（签名必须是 image_path: str，见文件头 FIX-3）
if VISION_TOOL_AVAILABLE:
    if hasattr(vision_detect, "name"):
        tools.append(vision_detect)
    else:
        @tool
        def detect_defect(image_path: str) -> str:
            """检测图片中的工业缺陷。输入图片路径，返回 JSON 格式的检测结果。"""
            return vision_detect(image_path)
        tools.append(detect_defect)
    print(f"[OK] 已加载精调视觉工具: {tools[-1].name}")
else:
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")

        @tool
        def detect_defect_fallback(image_path: str) -> str:
            """(兜底版) 使用 YOLOv8n 检测缺陷。"""
            results = model(image_path)
            return json.dumps([{"class": r.names[int(c)], "conf": float(conf)}
                               for r in results for c, conf in zip(r.boxes.cls, r.boxes.conf)])
        tools.append(detect_defect_fallback)
        print("[OK] 已加载内嵌兜底视觉工具: detect_defect_fallback")
    except Exception as e:
        print(f"[ERROR] 视觉工具完全不可用: {e}")

# 3.2 注册 RAG 工具
if RAG_TOOL_AVAILABLE:
    if hasattr(rag_search, "name"):
        tools.append(rag_search)
    else:
        @tool
        def search_maintenance_manual(query: str) -> str:
            """检索维修手册。当需要查询缺陷的处理流程、维修建议时调用。"""
            return rag_search(query)
        tools.append(search_maintenance_manual)
    print(f"[OK] 已加载 RAG 工具: {tools[-1].name}")

# ================= 4. Agent 状态定义 =================
class AgentState(TypedDict, total=False):
    # <<< FIX-1：Annotated 的第二个参数必须是可调用的 reducer（add），
    #            写字符串 "对话历史" 等于没有 reducer，messages 会被逐跳覆盖。
    messages: Annotated[Sequence[HumanMessage | SystemMessage | ToolMessage], add]
    user_id: str
    image_path: Optional[str]      # main.py 落盘后的真实绝对路径，全程不丢
    detect_result: Optional[str]   # 首次检测结论缓存，防止重复推理


def get_llm():
    """获取绑定工具的 LLM 实例"""
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.1,
    )
    return llm.bind_tools(tools)


def agent_node(state: AgentState):
    """Agent 决策节点"""
    llm = get_llm()

    tool_names = [t.name for t in tools]
    system_prompt = f"""你是一个工业缺陷检测专家助手。
当前可用工具：{tool_names}。

铁律（违反即视为任务失败）：
1. detect_defect 在整个会话中最多只调用一次；图片路径由系统自动注入，
   你不需要、也绝对不允许自己编造任何路径（如 /path/to/image.jpg、path_to_image）。
2. 一旦 tools 节点返回了缺陷清单（例如 inclusion 及其置信度、坐标），
   你的唯一任务就是立刻输出最终分析报告，禁止再调用任何工具。
3. 若 search_manual 返回"知识库服务暂时不可用"，直接采用其本地兜底建议，
   绝对禁止向用户索要图片或说"请上传图片"。
4. 报告必须包含：缺陷类型、置信度、位置坐标、影响评估、处置建议。

输出格式：中文分点报告，不要输出思考过程。
"""

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState):
    """工具执行节点 - 动态分发 + 路径强制注入 + 结果缓存"""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return {"messages": []}

    outputs = []
    updates = {}

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = dict(tool_call.get("args") or {})
        call_id = tool_call["id"]

        target_tool = next((t for t in tools if t.name == tool_name), None)
        if target_tool is None:
            outputs.append(ToolMessage(content=f"Tool not found: {tool_name}", tool_call_id=call_id))
            continue

        # ---------- 视觉工具：路径由系统接管 ----------
        if tool_name.startswith("detect_defect"):
            cached = state.get("detect_result")
            if cached:
                outputs.append(ToolMessage(content=cached, tool_call_id=call_id))
                print("[Tool] detect_defect 命中缓存，跳过重复推理（防幻觉二次调用）")
                continue

            ctx_path = _current_image_path.get() if _current_image_path is not None else None
            real_path = state.get("image_path") or ctx_path
            if not real_path:
                outputs.append(ToolMessage(
                    content="未获取到有效图片路径，请基于已有检测结果直接输出最终报告。",
                    tool_call_id=call_id))
                print("[WARN] tools_node 拿不到 image_path，已拦截本次工具调用")
                continue

            if tool_input.get("image_path") != real_path:
                print(f"[FIX] 覆盖大模型传入的幻觉路径 "
                      f"{tool_input.get('image_path')!r} -> 真实路径 {real_path}")
            tool_input["image_path"] = real_path          # <<< FIX-2 核心

        try:
            result = target_tool.invoke(tool_input)
            result_text = str(result)
            outputs.append(ToolMessage(content=result_text, tool_call_id=call_id))
            if tool_name.startswith("detect_defect") and "error" not in result_text.lower():
                updates["detect_result"] = result_text    # 缓存首次成功结果
            print(f"[Tool] 执行成功: {tool_name} -> {result_text[:60]}...")
        except Exception as e:
            outputs.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=call_id))
            print(f"[Tool] 执行失败: {tool_name} - {e}")

    updates["messages"] = outputs
    return updates


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    print("[OK] LangGraph Agent 使用内存模式构建完成")
    return app


# 全局单例
app = build_graph()

# ================= 6. API 调用封装 (供 main.py 使用) =================
def run_agent_stream(user_query: str, image_path: str = None, thread_id: str = "default_user"):
    """运行 Agent 并返回最终结果"""
    inputs = {
        "messages": [HumanMessage(content=user_query)],
        "user_id": thread_id,
        "image_path": image_path,      # <<< 直接进 State，不再往 messages 里塞路径文本
    }
    config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    for event in app.stream(inputs, config):
        for node_name, node_output in event.items():
            if node_name == "agent":
                msg = node_output["messages"][-1]
                if hasattr(msg, "content") and msg.content:
                    print(f"[Agent] {str(msg.content)[:50]}...")
            final_state = node_output

    if final_state and "messages" in final_state and final_state["messages"]:
        return final_state["messages"][-1].content
    return "Agent 未返回有效内容"


def get_app():
    """返回编译好的 LangGraph 应用单例（带 .stream()），供 main.py 初始化 Agent。"""
    return app


if __name__ == "__main__":
    print("=" * 50)
    print("开始本地测试 Agent (Thread: test_001)")
    result = run_agent_stream(
        user_query="帮我看看这张图有什么问题，并给出维修建议",
        image_path="test.jpg",
        thread_id="test_001",
    )
    print("\n" + "=" * 50)
    print(f"最终回复:\n{result}")