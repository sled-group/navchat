from multiprocessing import Queue
from typing import List
import gradio as gr
import cv2
from orion.agent_env.chatgpt_control_base import ChatGPTControlBase
from orion.agent_env.chatgpt_control_orion import ChatGPTControlORION
from orion.agent_env.chatgpt_control_cow import ChatGPTControlCoW
from orion.agent_env.chatgpt_control_vlmap import ChatGPTControlVLMap


END_SENT = "<eos>"
END_TURN = "<eot>"


class GradioInterface:
    def __init__(
        self,
        image_queue: Queue,
        user_message_queue: Queue,
        bot_message_queue: Queue,
    ):
        self.last_image = cv2.imread("orion/gradio_init_img.jpg")
        self.image_queue = image_queue
        self.user_message_queue = user_message_queue
        self.bot_message_queue = bot_message_queue

    def get_img(self):
        if self.image_queue.empty():
            return self.last_image
        else:
            self.last_image = self.image_queue.get()
            return self.last_image

    def process_user_message(self, user_message, history):
        self.user_message_queue.put(user_message)
        return "", history + [[user_message, None]]

    def process_bot_message(self, chat_history: List):
        chat_history.append([None, ""])
        bot_message_chuck: str = self.bot_message_queue.get()

        while bot_message_chuck != END_TURN:
            if bot_message_chuck == END_SENT:
                chat_history.append([None, ""])
            else:
                if "Command" in bot_message_chuck:
                    bot_message_chuck = bot_message_chuck.replace("Command", "Action")
                chat_history[-1][1] += bot_message_chuck
                yield chat_history

            bot_message_chuck = self.bot_message_queue.get()

    def run(self):
        with gr.Blocks(theme=gr.themes.Default(text_size="lg")) as gradio_demo:
            with gr.Column():
                with gr.Box():
                    gr.Markdown("## 🔥Navigation ChatBot Demo🚀")
            with gr.Row():
                with gr.Column(scale=1):
                    plot = gr.Image(self.last_image)
                with gr.Column(scale=2.5):
                    chatbot = gr.Chatbot()
                    chatbot.style(height=600)
                    msg = gr.Textbox()
                    msg.submit(
                        self.process_user_message,
                        [msg, chatbot],
                        [msg, chatbot],
                        show_progress=True,
                    ).then(self.process_bot_message, chatbot, chatbot)
            gradio_demo.load(self.get_img, None, plot, every=0.01)

        gradio_demo.queue().launch(
            server_name="127.0.0.1", server_port=7877, share=True
        )

class GradioDemoChatGPTControlORION(ChatGPTControlORION):
    def __init__(
        self, image_queue: Queue, user_message_queue: Queue, bot_message_queue: Queue,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_queue = image_queue
        self.user_message_queue = user_message_queue
        self.bot_message_queue = bot_message_queue

    def display(self, *args, **kwargs):
        super().display(*args, **kwargs)
        # make image smaller twice size for gradio
        self.display_image = cv2.resize(
            self.display_image, (self.display_image.shape[1] // 2, self.display_image.shape[0] // 2)
        )
        self.image_queue.put(cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB))

    def _get_user_input(self):
        user_input = self.user_message_queue.get()
        return user_input

    def _send_funcall_msg(self, msg):
        super()._send_funcall_msg(msg)
        self.bot_message_queue.put(
            "**API results**:  *" + msg.replace("\n", "<br>") + "*"
        )
        self.bot_message_queue.put(END_SENT)

    def _get_chatgpt_response(self):
        if self.use_stream:
            response = ""
            for chunk in self.chatgpt.get_system_response_stream():
                self.bot_message_queue.put(chunk.replace("\n", "<br>"))
                response += chunk
        else:
            response = self.chatgpt.get_system_response()
            self.bot_message_queue.put(response.replace("\n", "<br>"))
        self.bot_message_queue.put(END_SENT)
        return response

    def _post_process(self, command):
        super()._post_process(command)
        final_response = command["args"]["content"]
        self.bot_message_queue.put("**" + final_response + "**")
        self.bot_message_queue.put(END_TURN)