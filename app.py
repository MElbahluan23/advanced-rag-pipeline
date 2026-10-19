"""
Insurellm Expert Assistant — Gradio UI
----------------------------------------
Toggle between Simple and Pro RAG implementations via the dropdown.
Run: python app.py
"""

import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Lazy imports so switching mode doesn't crash if one DB is missing ──────────

def get_simple():
    from implementation.answer import answer_question
    return answer_question

def get_pro():
    from pro_implementation.answer import answer_question
    return answer_question


# ── Context formatter ──────────────────────────────────────────────────────────

def format_context(context, mode="simple"):
    color = "#ff7800" if mode == "simple" else "#6200ea"
    label = "Relevant Context" if mode == "simple" else "Relevant Context (Re-ranked)"
    result = f"<h2 style='color:{color};'>{label}</h2>\n\n"
    for i, doc in enumerate(context, 1):
        source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else doc.get("metadata", {}).get("source", "unknown")
        content = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
        result += f"<span style='color:{color};'>#{i} Source: {source}</span>\n\n"
        result += content + "\n\n"
    return result


# ── Gradio logic ───────────────────────────────────────────────────────────────

def chat(history, mode):
    last_message = history[-1]["content"]
    prior = history[:-1]

    answer_fn = get_simple() if mode == "Simple" else get_pro()
    answer, context = answer_fn(last_message, prior)

    history.append({"role": "assistant", "content": answer})
    return history, format_context(context, mode.lower())


def put_message(message, history):
    return "", history + [{"role": "user", "content": message}]


# ── UI ─────────────────────────────────────────────────────────────────────────

theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

with gr.Blocks(title="Insurellm Expert Assistant", theme=theme) as ui:
    gr.Markdown(
        "# 🏢 Insurellm Expert Assistant\n"
        "Ask me anything about Insurellm! Switch between RAG implementations using the dropdown."
    )

    with gr.Row():
        mode = gr.Dropdown(
            choices=["Simple", "Pro"],
            value="Simple",
            label="RAG Implementation",
            scale=1,
        )

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=580,
                type="messages",
                show_copy_button=True,
            )
            message = gr.Textbox(
                placeholder="Ask anything about Insurellm...",
                show_label=False,
            )

        with gr.Column(scale=1):
            context_display = gr.Markdown(
                label="📚 Retrieved Context",
                value="*Context will appear here after your first question*",
                container=True,
                height=580,
            )

    message.submit(
        put_message, [message, chatbot], [message, chatbot]
    ).then(chat, [chatbot, mode], [chatbot, context_display])


if __name__ == "__main__":
    ui.launch(inbrowser=True)
