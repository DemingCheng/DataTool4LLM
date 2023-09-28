import gradio as gr
import datetime
from transformers.utils.versions import require_version

require_version("gradio>=3.36.0", "To fix: pip install gradio>=3.36.0")


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Web Tuner") as demo:
        # top_elems = create_top()
        with gr.Tab("Train"):
            

        with gr.Tab("Evaluate"):
            

        with gr.Tab("Chat"):
            

        with gr.Tab("Export"):
            

        demo.load(
            manager.gen_label,
            [top_elems["lang"]],
            [elem for elems in elem_list for elem in elems.values()],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="10.60.10.4", server_port=7860, share=False, inbrowser=True)
