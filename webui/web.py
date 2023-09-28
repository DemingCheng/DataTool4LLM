from webui import create_ui

def create_ui():
    

def main():
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="10.60.10.4", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
