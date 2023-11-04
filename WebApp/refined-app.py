import os
from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import os
import re
import secrets
import tempfile
from modelscope import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from huggingface_hub import snapshot_download

# use qwen-VL as sample, to be replaced with your own VLM
DEFAULT_CKPT_PATH = '4bit/Qwen-VL-Chat-Int4'
REVISION = 'v1.0.0'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--revision", type=str, default=REVISION)
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    model_id = args.checkpoint_path
    model_dir = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        model_dir, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args, model, tokenizer):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        pic_idx = 1
        pre = ""
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                pre += q + '\n'
                pic_idx += 1
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        response, history = model.chat(tokenizer, message, history=history)
        image = tokenizer.draw_bbox_on_latest_picture(response, history)
        if image is not None:
            temp_dir = secrets.token_hex(20)
            temp_dir = Path(uploaded_file_dir) / temp_dir
            temp_dir.mkdir(exist_ok=True, parents=True)
            name = f"tmp{secrets.token_hex(5)}.jpg"
            filename = temp_dir / name
            image.save(str(filename))
            _chatbot[-1] = (_parse_text(chat_query), (str(filename),))
            chat_response = response.replace("<ref>", "")
            chat_response = chat_response.replace(r"</ref>", "")
            chat_response = re.sub(BOX_TAG_PATTERN, "", chat_response)
            if chat_response != "":
                _chatbot.append((None, chat_response))
        else:
            _chatbot[-1] = (_parse_text(chat_query), response)
        full_response = _parse_text(response)

        task_history[-1] = (query, full_response)
        print("VL-Chatbot: " + _parse_text(full_response))
        task_history = task_history[-10:]
        return _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        if file is not None:
          history = history + [((file.name,), None)]
          task_history = task_history + [((file.name,), None)]
        return history, task_history
    def add_files(history, task_history, files):
        if files is not None:
          for file in files:
              history = history + [((file.name,), None)]
              task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value=""),[]

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""\
<center><font size=3>This WebUI is a chatbot applicaton based on a VL-LLM model, fine tuned by the team especially on Foodie topic. \
</center>""")

        chatbot = gr.Chatbot(label='VL-Chat', elem_classes="control-height").style(height=300)
        task_history = gr.State([])
        with gr.Tab("Single Image") as tab_single:
            query = gr.Textbox(lines=2, label='Input')
            image = gr.File(file_types=["image"],interactive=True,label="Image File")
            with gr.Row():
                submit_btn = gr.Button("Submit")
                regen_btn = gr.Button("Retry")
                empty_btn = gr.Button("Reset Session")
            
            submit_btn.click(
              add_file, [chatbot, task_history, image], [chatbot, task_history], 
              show_progress=True
              ).then(
                add_text, [chatbot, task_history, query], [chatbot, task_history]
                ).then(
                  predict, [chatbot, task_history], [chatbot], show_progress=True
                  ).then(
                    reset_user_input, [], [query,image])
            empty_btn.click(reset_state, [task_history], [chatbot], show_progress=True)
            regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
            
        with gr.Tab("Batch with multiple images") as tab_batch:
            query = gr.Textbox(lines=2, label='Input')
            batch_images = gr.Files(file_count="multiple", file_types=["image"],interactive=True,label="Image Files")
            # addfile_btn = gr.UploadButton("📁 Upload Image File", file_types=["image"])
            with gr.Row():
                submit_btn = gr.Button("Submit")
                regen_btn = gr.Button("Retry")
                empty_btn = gr.Button("Reset Session")
            submit_btn.click(
              add_files, [chatbot, task_history, batch_images], [chatbot, task_history], 
              ).success(
                add_text, [chatbot, task_history, query], [chatbot, task_history]
                ).success(
                  predict, [chatbot, task_history], [chatbot], show_progress=True
                  ).then(
                    reset_user_input, [], [query,batch_images])
            empty_btn.click(reset_state, [task_history], [chatbot], show_progress=True)
            regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

        tab_single.select(reset_state, [task_history], [chatbot], show_progress=True)
        tab_batch.select(reset_state, [task_history], [chatbot], show_progress=True)

        gr.Markdown("""\
                <font size=2>Note: This quick demo is for school project Pattern Recognition system @NUS-ISS. \
                Not allowed for other usage. \
                Created in Oct 2023 """)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer)


if __name__ == '__main__':
    main()