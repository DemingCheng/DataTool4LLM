from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch


title = "ChatBot"
description = "Building a chatbot"
examples = [["How are you?"]]


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
# tokenizer = AutoTokenizer.from_pretrained("/data/intern/cmd/gpt2/project/model/mygpt2_batchsize_20")
# model = AutoModelForCausalLM.from_pretrained("/data/intern/cmd/gpt2/project/model/mygpt2_batchsize_20")

# def predict(input, history=[]):
#     # tokenize the new input sentence
#     new_user_input_ids = tokenizer.encode(
#         input + tokenizer.eos_token, return_tensors="pt"
#     )

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

#     # generate a response
#     history = model.generate(
#         bot_input_ids, max_length=4000, pad_token_id=tokenizer.eos_token_id
#     ).tolist()

#     # convert the tokens to text, and then split the responses into lines
#     response = tokenizer.decode(history[0]).split("<|endoftext|>")
#     # print('decoded_response-->>'+str(response))
#     response = [
#         (response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)
#     ]  # convert to tuples of list
#     # print('response-->>'+str(response))
#     return response, history

def predict(
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
):
    if text == "":
        yield chatbot, history, "Empty context."
        return

    inputs = generate_prompt_with_history(
        text, history, tokenizer, max_length=max_context_length_tokens
    )
    if inputs is None:
        yield chatbot, history, "Input too long."
        return
    else:
        prompt, inputs = inputs
        begin_length = len(prompt)
    input_ids = inputs["input_ids"][:, -max_context_length_tokens:].to(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for x in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=["User:", "Assistant:","</s>"],
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            if is_stop_word_or_prefix(x, ["User:", "Assistant:"]) is False:
                if "User:" in x:
                    x = x[: x.index("User:")].strip()
                if "Assistant:" in x:
                    x = x[: x.index("Assistant:")].strip()
                x = x.strip(" ")
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
                    [text, convert_to_markdown(x)]
                ], history + [[text, x]]
                yield a, b, "Generating..."
            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except:
                    pass
    torch.cuda.empty_cache()
    print(prompt)
    print(x)
    print("=" * 80)
    try:
        yield a, b, "Generate: Success"
    except:
        pass

def retry(
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(
        inputs,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
    ):
        yield x

def complete_fn(text, top_p, temperature, max_length):
    yield text
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda")
    with torch.no_grad():
        # gen_text = ""
        for x in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=["User:","</s>"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        ):
            # gen_text += x
            yield text+x

    torch.cuda.empty_cache()
    print(text+x)
    print("=" * 80)
    return text+x


with gr.Blocks(theme="finlaymacklon/boxy_violet") as demo: 
    with gr.Tab("文本补全模式"):
        with gr.Row(scale=1).style(equal_height=True):
            with gr.Column(min_width=50, scale=2):
                tb_notebook = gr.Textbox("User: who are you\nAssistant:", lines=80, maxlines=80, label="文本补全")
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# 设定参数")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        interactive=True,
                        label="最大采样概率阈值 (Top-p)",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="温度 (Temperature)",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=2048,
                        value=1024,
                        step=8,
                        interactive=True,
                        label="最大生成长度 (Max New Tokens)",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=128,
                        interactive=True,
                        label="最大历史长度 (Max History Tokens)",
                    )
                    btn_generate = gr.Button("生成")
                    btn_cancel = gr.Button("停止")
                    # gr.Examples(example_prompts, [tb_notebook],[tb_notebook])

        event_nb_submit = tb_notebook.submit(fn=complete_fn, inputs=[tb_notebook, top_p, temperature, max_length_tokens], outputs=[tb_notebook], show_progress=True)
        event_bt_click = btn_generate.click(fn=complete_fn, inputs=[tb_notebook, top_p, temperature, max_length_tokens], outputs=[tb_notebook], show_progress=True)
        btn_cancel.click(fn=lambda:None, cancels=[event_nb_submit,event_bt_click])
        
    retry_args = dict(
        fn=retry,
        inputs=[
            user_input,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

    # Chatbot
    cancelBtn.click(cancel_outputing, [], [status_display])
    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_question, user_input, submitBtn, cancelBtn],
        show_progress=True,
    )

    user_input.submit(**transfer_input_args).then(**predict_args)

    submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    retryBtn.click(**retry_args)

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )
demo.title = "Xiaoyan (小研)"

demo.queue().launch()

# gr.Interface(
#     fn=predict,
#     title=title,
#     description=description,
#     examples=examples,
#     inputs=["text", "state"],
#     outputs=["chatbot", "state"],
#     theme="finlaymacklon/boxy_violet",
# ).launch()