import gradio as gr
import requests
import json
import logging
import re
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}

stream_flag = True

DARK_CSS = """
/* ========== 全局基础 ========== */
body, .gradio-container {
    background: #0a0a12 !important;
    color: #c8d6e5 !important;
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
}

/* 隐藏 Gradio 默认 footer */
footer { display: none !important; }

/* ========== 主容器 ========== */
.gradio-container .contain {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

/* ========== 标题区域 ========== */
.gradio-container .prose h2 {
    color: #00e5ff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.3), 0 0 60px rgba(0, 229, 255, 0.1) !important;
    letter-spacing: 2px !important;
    border-bottom: 1px solid rgba(0, 229, 255, 0.2) !important;
    padding-bottom: 12px !important;
    margin-bottom: 8px !important;
}

.gradio-container .prose p {
    color: #7c8a9e !important;
    font-size: 0.95rem !important;
    opacity: 0.8 !important;
}

/* ========== Chatbot 聊天框 ========== */
.chatbot {
    background: #0d0d1a !important;
    border: 1px solid rgba(0, 229, 255, 0.15) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 30px rgba(0, 229, 255, 0.05), inset 0 0 60px rgba(0, 0, 0, 0.3) !important;
}

/* 用户消息气泡 */
.chatbot .user {
    background: linear-gradient(135deg, #132233 0%, #1a2a3a 100%) !important;
    border: 1px solid rgba(0, 229, 255, 0.25) !important;
    border-radius: 14px 14px 4px 14px !important;
    padding: 12px 18px !important;
    color: #d0e4f5 !important;
    box-shadow: 0 2px 12px rgba(0, 229, 255, 0.08) !important;
    margin: 8px 0 !important;
}

/* 助手消息气泡 */
.chatbot .bot {
    background: linear-gradient(135deg, #1a1028 0%, #1d1430 100%) !important;
    border: 1px solid rgba(168, 85, 247, 0.2) !important;
    border-radius: 14px 14px 14px 4px !important;
    padding: 12px 18px !important;
    color: #d8cfe8 !important;
    box-shadow: 0 2px 12px rgba(168, 85, 247, 0.06) !important;
    margin: 8px 0 !important;
}

/* Markdown 内容样式 */
.chatbot .message p { margin: 4px 0 !important; line-height: 1.65 !important; }
.chatbot .message strong { color: #00e5ff !important; }
.chatbot .message code {
    background: rgba(0, 229, 255, 0.1) !important;
    color: #00e5ff !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid rgba(0, 229, 255, 0.2) !important;
}

/* ========== 输入框 ========== */
.gradio-container textarea, .gradio-container input[type="text"] {
    background: #0d0d1a !important;
    border: 1px solid rgba(0, 229, 255, 0.2) !important;
    border-radius: 10px !important;
    color: #c8d6e5 !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    caret-color: #00e5ff !important;
}
.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.15) !important;
    outline: none !important;
}
.gradio-container textarea::placeholder, .gradio-container input::placeholder {
    color: #4a5568 !important;
}

/* ========== 发送按钮 ========== */
.gradio-container button.primary {
    background: linear-gradient(135deg, #006680 0%, #0088aa 50%, #006680 100%) !important;
    border: 1px solid rgba(0, 229, 255, 0.4) !important;
    border-radius: 10px !important;
    color: #e0f7ff !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-shadow: 0 0 8px rgba(0, 229, 255, 0.3) !important;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.15), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
.gradio-container button.primary:hover {
    background: linear-gradient(135deg, #0088aa 0%, #00aacc 50%, #0088aa 100%) !important;
    box-shadow: 0 0 24px rgba(0, 229, 255, 0.3), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transform: translateY(-1px) !important;
}
.gradio-container button.primary:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 8px rgba(0, 229, 255, 0.2) !important;
}

/* ========== 标签文字 ========== */
.gradio-container label, .gradio-container .label-text {
    color: #7c8a9e !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* ========== 滚动条 ========== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a12; }
::-webkit-scrollbar-thumb {
    background: rgba(0, 229, 255, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(0, 229, 255, 0.4); }

/* ========== 顶部扫描线效果（纯CSS） ========== */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
    animation: scanline 3s linear infinite;
    z-index: 9999;
    pointer-events: none;
    opacity: 0.5;
}
@keyframes scanline {
    0% { top: 0; }
    100% { top: 100%; }
}
"""


def send_message(user_message, history):
    data = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": "123",
        "conversationId": "123"
    }

    history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "正在生成回复..."}]
    yield history

    def format_response(full_text):
        formatted_text = full_text
        formatted_text = re.sub(r'<think>', '**思考过程**：\n', formatted_text)
        formatted_text = re.sub(r'</think>', '\n\n**最终回复**：\n', formatted_text)
        return formatted_text.strip()

    if stream_flag:
        assistant_response = ""
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    formatted_content = format_response(content)
                                    assistant_response += formatted_content
                                    yield history[:-1] + [{"role": "assistant", "content": assistant_response}]
                                if response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    break
                            except json.JSONDecodeError:
                                yield history[:-1] + [{"role": "assistant", "content": "解析响应时出错，请稍后再试。"}]
                                break
        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            yield history[:-1] + [{"role": "assistant", "content": "请求失败，请稍后再试。"}]
    else:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_json = response.json()
        assistant_content = response_json['choices'][0]['message']['content']
        formatted_content = format_response(assistant_content)
        yield history[:-1] + [{"role": "assistant", "content": formatted_content}]


with gr.Blocks(title="05_RagAgent_Business - 企业智能客服") as demo:
    gr.HTML("""
    <div style="text-align:center; margin-bottom:10px;">
        <h1 style="
            color:#00e5ff;
            font-size:2.2rem;
            letter-spacing:6px;
            text-shadow:0 0 30px rgba(0,229,255,0.4), 0 0 80px rgba(0,229,255,0.15);
            margin:0 0 6px 0;
            font-weight:800;
        ">◆ 企业智能客服 ◆</h1>
        <p style="
            color:#5a6a80;
            font-size:0.85rem;
            letter-spacing:3px;
            margin:0;
        ">ENTERPRISE SMART CUSTOMER SERVICE</p>
        <div style="
            width:60px; height:2px;
            background: linear-gradient(90deg, transparent, #00e5ff, transparent);
            margin: 12px auto;
        "></div>
    </div>
    """)

    chatbot = gr.Chatbot(label="", height=480)

    with gr.Row():
        message = gr.Textbox(
            label="",
            placeholder="> 输入问题，智能客服即刻响应...",
            scale=8,
            show_label=False,
        )
        send = gr.Button("▸ 执行", scale=2, variant="primary")

    gr.HTML("""
    <div style="text-align:center; margin-top:16px; opacity:0.4;">
        <span style="color:#00e5ff; font-size:0.75rem; letter-spacing:2px;">
            ⬡ QWEN2.5:7B &nbsp;|&nbsp; ⬡ NOMIC-EMBED-TEXT &nbsp;|&nbsp; ⬡ CHROMADB &nbsp;|&nbsp; ⬡ LANGGRAPH
        </span>
    </div>
    """)

    send.click(send_message, [message, chatbot], chatbot)
    message.submit(send_message, [message, chatbot], chatbot)
    send.click(lambda: "", None, message)
    message.submit(lambda: "", None, message)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=DARK_CSS,
    )
