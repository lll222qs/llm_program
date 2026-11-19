
# main.py
# FastAPI 主应用入口 - 完整版（支持文件上传 + Base64 + 图片压缩）
# 修复重点：/docs 补上 swagger_ui_standalone_preset_url，彻底解决 Swagger 白屏
from agent_day2_vision import _current_image_path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import uvicorn
from typing import Optional, List
import base64
import os
import traceback
from contextlib import asynccontextmanager
from PIL import Image
import io
import shutil


# 导入配置 (确保 config.py 在 seven_week4 目录下，或者调整导入路径)
try:
    from config import settings
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import settings

# 导入 LangGraph 工厂函数
try:
    from agent_day2_graph import get_app
except ImportError as e:
    print(f"警告: 导入 get_app 失败: {e}")   # 现在能区分"文件没找到"还是"函数名不对"
    get_app = None

from langchain_core.messages import HumanMessage
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# ==========================================
# 1. 图片处理工具函数
# ==========================================

def resize_image_base64(base64_str: str, max_size: int = 800, quality: int = 85) -> str:
    """将 Base64 图片缩放压缩，避免显存爆炸导致卡死"""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  [图片] 已缩放至: {new_size}")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  [图片] 压缩失败，使用原图: {e}")
        return base64_str


def file_to_base64(file_path: str) -> str:
    """读取本地文件并转为 Base64"""
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def construct_multimodal_message(image_b64: str, prompt: str = "请分析这张图片。") -> list:
    """构造符合 OpenAI/Ollama 标准的图片消息格式"""
    compressed_b64 = resize_image_base64(image_b64)
    image_url = f"data:image/jpeg;base64,{compressed_b64}"
    return [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt}
    ]


def extract_response_from_events(all_events: list) -> tuple:
    """从 LangGraph 事件中提取响应内容和节点轨迹"""
    response_content = ""
    node_trace = []

    for event in all_events:
        for node_name, node_output in event.items():
            node_trace.append(f"[{node_name}]")
            if isinstance(node_output, dict) and "messages" in node_output:
                messages = node_output["messages"]
                for msg in messages:
                    if hasattr(msg, "content"):
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            text_parts = [
                                p.get("text", "") for p in msg_content
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            msg_content = "".join(text_parts)
                        if msg_content and len(str(msg_content)) > 10:
                            node_trace.append(f"  ↳ {str(msg_content)[:100]}...")

    # 优先从最后一个状态提取
    last_event = all_events[-1] if all_events else None
    if last_event:
        for node_name, node_output in last_event.items():
            if isinstance(node_output, dict) and "messages" in node_output:
                msgs = node_output["messages"]
                if msgs:
                    last_msg = msgs[-1]
                    raw_content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    if isinstance(raw_content, list):
                        text_parts = [
                            p.get("text", "") for p in raw_content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        response_content = "".join(text_parts)
                    else:
                        response_content = raw_content

    # 兜底：尝试找 'agent' 节点
    if not response_content:
        for event in all_events:
            if "agent" in event:
                agent_out = event["agent"]
                if isinstance(agent_out, dict) and "messages" in agent_out:
                    msgs = agent_out["messages"]
                    if msgs:
                        raw_content = msgs[-1].content
                        if isinstance(raw_content, list):
                            text_parts = [
                                p.get("text", "") for p in raw_content
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            response_content = "".join(text_parts)
                        else:
                            response_content = raw_content

    if not response_content:
        response_content = "⚠️ Agent 运行结束，但未生成有效回复。请检查后台日志。"

    return response_content, node_trace


# ==========================================
# 2. 生命周期管理
# ==========================================
agent_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app
    print("=" * 50)
    print("🚀 正在初始化 LangGraph Agent...")
    try:
        if get_app:
            agent_app = get_app()
            print("✅ LangGraph Agent 初始化成功")
        else:
            print("❌ get_app 函数未找到")
    except Exception as e:
        print(f"❌ LangGraph 初始化失败: {e}")
        traceback.print_exc()
        agent_app = None
    yield
    print("🛑 应用关闭，清理资源...")

# ==========================================
# 3. 初始化 FastAPI
# ==========================================
app = FastAPI(
    title="工业视觉质检 Agent",
    description="基于 Agentic RAG 的工业视觉质检与运维系统",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan
)

# ---- 自定义文档路由：必须覆盖三个外部资源，否则白屏 ----
# 之前只覆盖了 bundle.js 和 css，漏了真正驱动渲染的 standalone-preset.js
from fastapi.responses import HTMLResponse

# ... 其他代码保持不变 ...

from fastapi.responses import HTMLResponse

# ... 其他代码保持不变 ...

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    # 使用 unpkg 源，并指定 5.17.14 版本（这个版本非常稳定）
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LangGraph Agent - Swagger UI</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <!-- 引入 CSS -->
            <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css">
            <!-- 引入 Favicon -->
            <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5.17.14/favicon-32x32.png">
        </head>
        <body>
            <div id="swagger-ui"></div>
            <!-- 引入 JS -->
            <script src="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
            <script src="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-standalone-preset.js"></script>
            
            <script>
                window.onload = function() {
                    const ui = SwaggerUIBundle({
                        url: '/openapi.json', // 告诉它去哪里获取接口文档数据
                        dom_id: '#swagger-ui',
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIStandalonePreset // 启用独立预设
                        ],
                        layout: "StandaloneLayout"
                    });
                };
            </script>
        </body>
        </html>
        """
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.bootcdn.net/ajax/libs/redoc/2.1.4/redoc.standalone.js",
    )

# ==========================================
# 4. 数据模型定义
# ==========================================
class ImageRequest(BaseModel):
    """请求体：支持本地路径或 Base64"""
    image_input: str = Field(..., description="图片本地路径 (如 D:/img.jpg) 或 Base64 字符串")
    thread_id: Optional[str] = Field(default="default-session", description="会话ID，用于保持上下文")

class DetectionResponse(BaseModel):
    """响应体"""
    status: str
    report: str
    thread_id: str
    trace: List[str] = []

# ==========================================
# 5. 核心接口 A: Base64/路径模式
# ==========================================
@app.post("/detect_and_advice", response_model=DetectionResponse)
async def detect_and_advice(request: ImageRequest):
    """
    图片检测与建议接口（Base64/路径模式）：
    1. 接收图片（路径或Base64）
    2. 转换为标准多模态格式
    3. 调用 LangGraph 进行分析
    4. 返回分析报告与执行轨迹
    """
    if not agent_app:
        raise HTTPException(
            status_code=503,
            detail="Agent 尚未初始化完成，请检查上游服务（Redis/Ollama/Qdrant）是否正常运行"
        )

    try:
        # --- A. 智能处理输入 ---
        image_b64 = ""

        if os.path.exists(request.image_input):
            print(f"📂 检测到本地图片路径: {request.image_input}")
            image_b64 = file_to_base64(request.image_input)

        elif request.image_input.startswith("data:image"):
            pure_b64 = request.image_input.split(",", 1)[1]
            image_b64 = pure_b64
            print(f"🖼️ 检测到 Base64 图片数据 (已去前缀)")

        elif len(request.image_input) > 100 and not request.image_input.startswith("http"):
            image_b64 = request.image_input
            print("🖼️ 检测到纯 Base64 数据")

        else:
            print(f"📝 当作普通文本处理: {request.image_input[:30]}...")
            content_to_send = request.image_input
            inputs = {"messages": [HumanMessage(content=content_to_send)]}
            config = {"configurable": {"thread_id": request.thread_id}}
            print(f"⏳ 开始处理会话: {request.thread_id}")

            all_events = []
            for event in agent_app.stream(inputs, config=config):
                all_events.append(event)
                print(f"   -> 节点执行: {list(event.keys())}")

            response_content, node_trace = extract_response_from_events(all_events)
            print(f"\n✅ 最终回复预览: {str(response_content)[:100]}...")
            return {
                "status": "success",
                "report": response_content,
                "thread_id": request.thread_id,
                "trace": node_trace[-15:]
            }

        # --- B. 构造多模态消息（图片模式）---
        print(f"🖼️ 图片 Base64 长度: {len(image_b64)}")

        import uuid

        # 1. 把 base64 图片真正写入容器内的文件
        os.makedirs("/app/temp_images", exist_ok=True)
        real_path = os.path.join("/app/temp_images", f"{uuid.uuid4().hex}.jpg") # 使用绝对路径
        with open(real_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        print(f"💾 图片已落盘: {real_path}")

        # 2. 核心：将真实路径注入到 agent_day2_vision 的全局上下文中
        _current_image_path.set(real_path)
        print(f"[🚀] 已将真实路径 {real_path} 注入到视觉工具上下文")

        # 3. 正常构造多模态消息（这部分保持你原来的不动）
        message_content = construct_multimodal_message(image_b64)

        # --- C. 运行 LangGraph ---
        import contextvars
        # 假设你已经在 agent_day2_vision.py 的顶部加了这行全局变量：
        # _current_image_path = contextvars.ContextVar("current_image_path", default=None)

        # 1. 设置上下文里的真实图片路径
        _current_image_path.set(real_path)

        # 2. 保持你原有的多模态构造逻辑不变（模型看图用这个）
        message_content = construct_multimodal_message(image_b64)

        # 3. 核心修改：在 inputs 的字典里加入真实路径，作为全局上下文
        # LangGraph 在执行 tools_node 时，可以通过 state 访问到它
        inputs = {
            "messages": [HumanMessage(content=message_content)],
            "image_path": real_path  
        }
        config = {"configurable": {"thread_id": request.thread_id}}
        print(f"⏳ 开始处理会话: {request.thread_id}")

        all_events = []
        for event in agent_app.stream(inputs, config=config):
            all_events.append(event)
            print(f"   -> 节点执行: {list(event.keys())}")

        # --- D. 提取结果 ---
        response_content, node_trace = extract_response_from_events(all_events)
        print(f"\n✅ 最终回复预览: {str(response_content)[:100]}...")

        return {
            "status": "success",
            "report": response_content,
            "thread_id": request.thread_id,
            "trace": node_trace[-15:]
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"图片文件未找到: {request.image_input}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")


# ==========================================
# 6. 核心接口 B: 直接文件上传模式（推荐）
# ==========================================
@app.post("/detect_and_advice_file", response_model=DetectionResponse)
async def detect_and_advice_file(
    file: UploadFile = File(...),
    thread_id: str = Form(default="default-session")
):
    """
    直接接收图片文件上传的接口（推荐方式）
    无需手动转 Base64，直接在 Swagger 页面点击 'Choose File' 上传即可
    """
    if not agent_app:
        raise HTTPException(
            status_code=503,
            detail="Agent 尚未初始化完成，请检查上游服务（Redis/Ollama/Qdrant）是否正常运行"
        )

    temp_path = None
    try:
        # 1. 保存临时文件
        temp_path = f"./temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(temp_path)
        print(f"📂 接收到文件: {file.filename}, 大小: {file_size} bytes")

        # 2. 读取文件转 Base64
        image_b64 = file_to_base64(temp_path)
        print(f"🖼️ 图片 Base64 长度: {len(image_b64)}")

        # 3. 构造多模态消息
        message_content = construct_multimodal_message(image_b64)
        print("✅ 消息已封装为多模态格式 (Text + Image)")

        # 4. 运行 LangGraph
        inputs = {"messages": [HumanMessage(content=message_content)]}
        config = {"configurable": {"thread_id": thread_id}}
        print(f"⏳ 开始处理会话: {thread_id}")

        all_events = []
        for event in agent_app.stream(inputs, config=config):
            all_events.append(event)
            print(f"   -> 节点执行: {list(event.keys())}")

        # 5. 提取结果
        response_content, node_trace = extract_response_from_events(all_events)
        print(f"\n✅ 最终回复预览: {str(response_content)[:100]}...")

        return {
            "status": "success",
            "report": response_content,
            "thread_id": thread_id,
            "trace": node_trace[-15:]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"🧹 已清理临时文件: {temp_path}")

# ==========================================
# 7. 启动入口
# ==========================================
if __name__ == "__main__":
    port = getattr(settings, "AGENT_PORT", 8000)
    print(f"🔥 正在启动服务，端口: {port} ...")
    print(f"📍 当前运行文件: {os.path.abspath(__file__)}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )