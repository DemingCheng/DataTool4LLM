import gradio as gr
import random
import time
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("/data/intern/cmd/gpt2/project/model/mygpt2_batchsize_15")
model = AutoModelForCausalLM.from_pretrained("/data/intern/cmd/gpt2/project/model/mygpt2_batchsize_15", pad_token_id=tokenizer.eos_token_id)
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


# def predict(input, history):
#     input = 'Senior: '+ input
#     text = input + ' Caregiver: '
#     token = tokenizer(input, return_tensors="pt")
#     # streamer = TextStreamer(tokenizer)
#     output = model.generate(
#         **token, 
#         # streamer=streamer, 
#         max_new_tokens=20, 
#         no_repeat_ngram_size=2,
#         early_stopping=True,
#         temperature=0.5,
#         top_p=0.5
#     )
#     output = tokenizer.decode(output[0], skip_special_tokens=True)
#     history = history + [[input, output]]
#     time.sleep(2)
#     return "", history

def train(epoch, bath_size):
    raw_data = load_dataset("text",data_files="/data/intern/cmd/gpt2/project/train_data.txt")

    model = AutoModelForCausalLM.from_pretrained("gpt2") 
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_size="left")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(raw_data):
        tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_size="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(raw_data["text"], padding=True, truncation=True)

    tokenized_datasets = raw_data.map(tokenize_function, batched=True)
    print(tokenized_datasets)

    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

    training_args = TrainingArguments(
        
        output_dir='./results', # 输出目录
        num_train_epochs=int(epoch),              # 训练轮数
        per_device_train_batch_size=int(bath_size),   # 每个设备上的训练批次大小
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=1,
        save_steps=5, 
        # warmup_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer
    )
    trainer.train()
    
    # print("trainer.state.log_history: ", trainer.state.log_history)
    trainer.save_model(output_dir = "./model/mygpt2_batchsize_32")
    # loss_curve(trainer.state.log_history)
    return trainer.state.log_history

def output(log_history):
    #分别存储神经网络训练的step和loss值
    step = [item['step'] for item in log_history]
    # loss = [item['loss'] for item in data]
    loss = [item['loss'] if 'loss' in item else None for item in log_history]
    return step, loss

def predict(input, max_length, top_p, temperature, history):
    input = 'Senior: '+ input
    text = input + ' Caregiver: '
    history_input = ' '.join(history)
    print(history_input)
    token = tokenizer(input, return_tensors="pt")
    # streamer = TextStreamer(tokenizer)
    output = model.generate(
        **token, 
        # streamer=streamer, 
        max_new_tokens=max_length, 
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=temperature,
        top_p=top_p
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)
    print('Caregiver: ')
    history = history + [[input, 'Caregiver: ' + output]]
    time.sleep(2)
    return "", history


with gr.Blocks() as demo:
    with gr.Tab("Train"):
        with gr.Row():
            with gr.Column(scale=4):
                train_log = gr.Textbox(label="train_log")
            with gr.Column(scale=1):
                epoch = gr.Number(label="epoch")
                batch_size = gr.Number(label="batch_size")
                btn_train = gr.Button("Train")

    btn_train.click(fn = train, inputs = [epoch, batch_size], outputs = train_log).then()

    with gr.Tab("Test"):
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot()
                msg = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=11).style(container=False)
            with gr.Column(scale=1):
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
                btn_generate = gr.Button("Generate")
                btn_clear = gr.ClearButton([msg, chatbot])

    # msg.submit(respond, [msg, chatbot], [msg, chatbot])
    # msg.submit(predict, [msg, max_length, top_p, temperature, chatbot], [msg, chatbot])
    btn_generate.click(predict, [msg, max_length, top_p, temperature, chatbot], [msg, chatbot])
    
demo.launch()