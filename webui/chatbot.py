import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from typing import Any, Dict, Generator, List, Optional, Tuple

# class Dataset():
#     def __init__(self):
#         pass
    
#     def load_dataset():
#         pass

#     # Info Function
#     def preview_dataset():
#         pass

#     def calculate_tokens():
#         pass

#     def calculate_size():
#         pass
    
#     # Process Function
#     def filter():
#         pass
    
#     # Generate Function
#     def self_cognition():
#         pass

class ChatModel():
    def __init__(self, args:Optional[Dict[str, Any]] = None):
        self.generating_args = dict(
            max_length = 512,
            max_new_tokens = 512,
            temperature=0.5,
            top_p = 0.5,
            top_k = 3,
            early_stopping = True
        )

    def load_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",trust_remote_code=True).eval()
        # self.tokenizer.padding_side = "left"
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        # return self.model, self.tokenizer, self.model.generation_config
        return "Successfully loaded model:" + model_path
    
    def load_config(self, model_config:Optional[Dict[str, Any]] = None):
        self.generating_args['max_length'] = model_config['max_length']


    def infer(self, message, history):
        response, history = self.model.chat(self.tokenizer, message, history=history)
        history.append((message, response))
        return "", history
    
    def infer(
        self, 
        query:str, 
        history:List[Tuple[str, str]]
    ):
        # tokenized_input = self.tokenizer(query, padding = True, truncation = True,return_tensors = 'pt')
        tokenized_input = self.tokenizer(query, return_tensors = 'pt')
        generation_config = GenerationConfig(
            max_length = self.generating_args['max_length'],
            max_new_tokens = self.generating_args['max_new_tokens'],
            temperature = self.generating_args['temperature'],
            early_stopping = self.generating_args['early_stopping']
        )
        
        tokenized_input = tokenized_input.to(self.model.device)
        outputs = self.model.generate(**tokenized_input, generation_config=generation_config)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        history.append((query, response))
        return "", history

    def respond(message, chat_history):
        bot_message = "How are you?"
        chat_history.append((message, bot_message))
        return "", chat_history
    
    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = "How are you?"
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            yield history
  
def creat_webui() -> gr.Blocks:
    with gr.Blocks(title="Web QA") as demo:
        # top_elems = create_top()
        with gr.Tab("Chat"):
            with gr.Row():
                chat_model = ChatModel()
                with gr.Column(scale=4):
                    # model_name = gr.Dropdown(choices=["Qwen-7B", "LLaMA-7B"], scale=3)
                    model_path = gr.Textbox(label="Model Path", value="/data/zxu/Qwen-7B-Chat", placeholder="/data/zxu/Qwen-7B-Chat")
                    load_status = gr.Textbox(label="Load Status", placeholder="No model is Loaded")
                with gr.Column(scale=1):
                    load_btn = gr.Button(value="Load")
                    unload_btn = gr.Button(value="Unload")

                load_btn.click(
                    fn=chat_model.load_model,
                    inputs=[model_path],
                    outputs=load_status
                )

            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox(show_label=False, lines=8)

                with gr.Column(scale=1):
                    max_new_tokens_box = gr.Slider(
                        minimum=10, 
                        maximum=2048, 
                        label='Max New Tokens',
                        value=chat_model.generating_args['max_new_tokens'],
                        step=1,
                        interactive=True
                    )
                    temperature_box = gr.Slider(
                        minimum=0.01, 
                        maximum=1, 
                        label='Temperature',
                        value=chat_model.generating_args['temperature'], 
                        step=0.01,
                        interactive=True
                        # visible=False
                    )
                    top_p_box = gr.Slider(
                        minimum=0.01, 
                        maximum=1, 
                        label='Top-p',
                        value=chat_model.generating_args['top_p'], 
                        step=0.01,
                        interactive=True
                    )
                    top_k_box = gr.Slider(
                        minimum=0.01, 
                        maximum=1, 
                        label='Top-k',
                        value=chat_model.generating_args['top_k'], 
                        step=0.01,
                        interactive=True
                    )
                    early_stopping_box = gr.Radio(
                        choices=["True","False"],
                        value=str(chat_model.generating_args['early_stopping']),
                        label='early_stopping',
                        interactive=True
                    )
                 
                    submit_btn = gr.Button(value="Submit")
                    clear_btn = gr.ClearButton([msg, chatbot])
                    
                submit_btn.click(chat_model.infer, [msg, chatbot], [msg, chatbot])
                msg.submit(chat_model.infer, [msg, chatbot], [msg, chatbot])
                # clear_btn.click(lambda: ([], []), outputs=[msg, chatbot], show_progress=True)

        with gr.Tab("Data"):
            pass

        with gr.Tab("Train"):
            pass

        with gr.Tab("Evaluate"):
            pass

        with gr.Tab("Export"):
            pass

        # demo.load(
        #     predict,
        #     [message, history],
        #     [, history]
        # )

    return demo


def main():
    demo = creat_webui()
    demo.queue()
    demo.launch(server_name="10.60.10.4", server_port=7860, share=False, inbrowser=True)

if __name__ == "__main__":
    main()
