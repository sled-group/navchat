import random

import gradio as gr

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()

    def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860, debug=True)
