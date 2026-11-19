
import cv2
import numpy as np
import onnxruntime as ort
from langchain_core.tools import tool
import os
import json
import contextvars
_current_image_path = contextvars.ContextVar("current_image_path", default=None)

# ================= 配置区 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "runs", "detect", "neu_det_runs", "exp1", "weights", "best.onnx")

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
NUM_CLASSES = len(CLASS_NAMES)  # 6
EXPECTED_DIM = 4 + NUM_CLASSES   # 10 ← 关键修正！
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

print(f"[START] [Vision Agent] 模型路径: {MODEL_PATH}")
print(f"[INFO] [Vision Agent] 期望输出维度: {EXPECTED_DIM} (4 box + {NUM_CLASSES} classes)")

# ================= 懒加载模型 =================
_session = None
_input_name = None

def get_session():
    global _session, _input_name
    if _session is None:
        print(f"[LOAD] [Vision Agent] 首次调用，正在加载模型...")
        if not os.path.exists(MODEL_PATH):
            print(f"[ERR] [Vision Agent] 模型文件不存在: {MODEL_PATH}")
            print(f"📂 [Debug] 当前目录: {os.listdir(current_dir)}")
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        
        _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        _input_name = _session.get_inputs()[0].name
        
        # 打印模型实际输出形状，一劳永逸确认格式
        out_shape = _session.get_outputs()[0].shape
        print(f"[OK] [Vision Agent] 模型加载成功！输出形状: {out_shape}")
    return _session, _input_name

# ================= 预处理 =================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    original_h, original_w = img.shape[:2]
    img_lb, ratio, (dw, dh) = letterbox(img, (IMG_SIZE, IMG_SIZE))
    
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    input_tensor = np.expand_dims(img_transposed, axis=0)
    
    print(f"🖼️ [Debug] 原图: {original_w}x{original_h}, ratio={ratio:.4f}, pad=({dw:.1f},{dh:.1f})")
    return input_tensor, (original_h, original_w), ratio, (dw, dh)

# ================= IoU 计算 =================
def calculate_iou(box_a, boxes_b):
    def cxywh_to_xyxy(cxywh):
        cx, cy, w, h = cxywh
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    a_xyxy = cxywh_to_xyxy(box_a)
    b_xyxy = np.array([cxywh_to_xyxy(b) for b in boxes_b])
    
    inter_x1 = np.maximum(a_xyxy[0], b_xyxy[:, 0])
    inter_y1 = np.maximum(a_xyxy[1], b_xyxy[:, 1])
    inter_x2 = np.minimum(a_xyxy[2], b_xyxy[:, 2])
    inter_y2 = np.minimum(a_xyxy[3], b_xyxy[:, 3])
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    area_a = (a_xyxy[2] - a_xyxy[0]) * (a_xyxy[3] - a_xyxy[1])
    area_b = (b_xyxy[:, 2] - b_xyxy[:, 0]) * (b_xyxy[:, 3] - b_xyxy[:, 1])
    union_area = area_a + area_b - inter_area
    
    return inter_area / (union_area + 1e-7)

# ================= 后处理（核心修复）=================
def postprocess_detections(outputs, original_shape, ratio, pad):
    raw_output = outputs[0]
    predictions = np.squeeze(raw_output)  # 去掉 batch 维度
    
    print(f"📐 [Debug] squeeze后形状: {predictions.shape}")
    
    # 【修复】用 EXPECTED_DIM(10) 代替硬编码的 84
    if predictions.ndim == 2:
        if predictions.shape[0] == EXPECTED_DIM:
            # (10, 8400) -> 转置为 (8400, 10)
            predictions = predictions.T
            print(f"[LOAD] [Debug] 检测到 ({EXPECTED_DIM}, N) 格式，已转置为 (N, {EXPECTED_DIM})")
        elif predictions.shape[1] == EXPECTED_DIM:
            # 已是 (8400, 10)，无需转置
            print(f"[OK] [Debug] 已是 (N, {EXPECTED_DIM}) 格式，直接使用")
        else:
            raise ValueError(
                f"[ERR] 输出维度不匹配！期望第二维为{EXPECTED_DIM}，"
                f"实际形状: {predictions.shape}。"
                f"请确认模型类别数是否为{NUM_CLASSES}"
            )
    else:
        raise ValueError(f"[ERR] 意外的输出维度: {predictions.ndim}D, shape={predictions.shape}")
    
    # 此时 predictions 一定是 (8400, 10)
    boxes = predictions[:, :4]       # (8400, 4) cx,cy,w,h
    scores = predictions[:, 4:]      # (8400, 6) 各类别分数
    
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # 详细调试日志
    print(f"[SHAPE] [Debug] 分数统计: min={max_scores.min():.4f}, max={max_scores.max():.4f}, mean={max_scores.mean():.4f}")
    
    # Top5 诊断
    top5_idx = np.argsort(max_scores)[::-1][:5]
    for i, idx in enumerate(top5_idx):
        cid = class_ids[idx]
        cls_name = CLASS_NAMES[cid] if cid < NUM_CLASSES else f"UNKNOWN({cid})"
        print(f"   Top{i+1}: score={max_scores[idx]:.4f}, class={cls_name}(id={cid}), box={boxes[idx]}")
    
    # 阈值过滤
    mask = max_scores > CONF_THRESHOLD
    pre_nms_count = int(np.sum(mask))
    print(f"🔢 [Debug] 超过阈值({CONF_THRESHOLD}): {pre_nms_count}/{len(max_scores)}")
    
    filtered_boxes = boxes[mask]
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]
    
    if pre_nms_count == 0:
        print("[WARN] [Debug] 无框通过阈值过滤")
        return []

    # NMS
    indices = np.argsort(filtered_scores)[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        current_box = filtered_boxes[current]
        remaining_indices = indices[1:]
        remaining_boxes = filtered_boxes[remaining_indices]
        ious = calculate_iou(current_box, remaining_boxes)
        low_iou_indices = np.where(ious < IOU_THRESHOLD)[0]
        indices = remaining_indices[low_iou_indices]
    
    print(f"🏁 [Debug] NMS后保留: {len(keep)} 个框")

    # Letterbox 坐标还原
    final_results = []
    orig_h, orig_w = original_shape
    dw, dh = pad
    
    for idx in keep:
        cx, cy, w, h = filtered_boxes[idx]
        x1 = int((cx - w/2 - dw) / ratio)
        y1 = int((cy - h/2 - dh) / ratio)
        x2 = int((cx + w/2 - dw) / ratio)
        y2 = int((cy + h/2 - dh) / ratio)
        
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))
        
        cid = int(filtered_class_ids[idx])
        final_results.append({
            "box": [x1, y1, x2, y2],
            "conf": float(filtered_scores[idx]),
            "cls_id": cid,
            "cls_name": CLASS_NAMES[cid]
        })
    return final_results


# 修改函数的入参，传入整个 state
@tool
def detect_defect(image_path: str) -> str:
    """检测工业钢材表面缺陷。输入 state 字典，获取 image_path，返回缺陷类型、置信度和位置。"""
    
    # 1. 从 state 里拿出真实的图片路径
    # 如果 state 里没有 image_path，说明 agent 在乱调用，直接返回错误
    _current_image_path.get()
    
    print(f"👁️ [Vision Agent] 准备从 State 读取路径进行检测...")

    # 2. 校验路径有效性
    if not image_path or not os.path.exists(image_path):
        error_msg = "无法获取有效的图片路径，请检查输入。"
        print(f"[WARN] [Vision Agent] 路径无效或不存在: {image_path}")
        return error_msg

    # 3. 真正的检测开始
    print(f"👁️ [Vision Agent] 读取到真实路径: {image_path}，开始检测")
    
    try:
        session, input_name = get_session()
        input_tensor, original_shape, ratio, (dw, dh) = preprocess_image(image_path)
        
        outputs = session.run(None, {input_name: input_tensor})
        
        results = postprocess_detections(outputs, original_shape, ratio, (dw, dh))
        
        if not results:
            print("[WARN] [Vision Agent] 未检测到明显缺陷。")
            return "未检测到明显的表面缺陷。"
            
        print(f"[OK] [Vision Agent] 检测到 {len(results)} 个缺陷！")
        report = []
        for det in results:
            x1, y1, x2, y2 = det['box']
            report.append(
                f"- 发现缺陷: {det['cls_name']} "
                f"(置信度: {det['conf']:.2f}, 位置: [{x1},{y1}] -> [{x2},{y2}])"
            )
        return "\n".join(report)
        
    except Exception as e:
        import traceback
        print(f"[ERR] [Vision Agent] 检测出错: {e}")
        traceback.print_exc()
        return f"检测失败: {str(e)}"

if __name__ == "__main__":
    test_img = "/app/data/test.jpg"
    if os.path.exists(test_img):
        print(detect_defect.invoke(test_img))
    else:
        print(f"测试图片不存在: {test_img}，跳过测试。")