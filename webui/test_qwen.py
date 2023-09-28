from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/data/zxu/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/data/zxu/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("/data/zxu/Qwen-7B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

response, history = model.chat(tokenizer, "你好", history=None)
print(response)

response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)

response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)




import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def predict(inp):
    input_ids = tokenizer.encode(inp, return_tensors='pt')
    beam_output = model.generate(input_ids, max_length=100, num_beams=5,
                                 no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."  # so that max_length doesn't stop an output abruptly.


INPUTS = gr.inputs.Textbox()
OUTPUTS = gr.outputs.Textbox()
gr.Interface(
    fn=predict, 
    inputs=INPUTS, 
    outputs=OUTPUTS, 
    title="GPT-2",
    description="Building a chatbot",
    capture_session=False,
    theme="finlaymacklon/boxy_violet"
).launch(inbrowser=True)

    