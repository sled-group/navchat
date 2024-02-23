from multiprocessing import Process, Queue

from orion.utils.gradio_interface import (
    GradioInterface,
    GradioDemoChatGPTControlORION,
)
from orion.config.chatgpt_config import AzureGPT4Config


def run_gradio(image_queue, user_message_queue, bot_message_queue):
    gradio_interface = GradioInterface(
        image_queue=image_queue,
        user_message_queue=user_message_queue,
        bot_message_queue=bot_message_queue,
    )
    gradio_interface.run()


def main():
    user_message_queue = Queue()
    bot_message_queue = Queue()
    image_queue = Queue()

    p = Process(
        target=run_gradio, args=(image_queue, user_message_queue, bot_message_queue)
    )
    p.start()

    game = GradioDemoChatGPTControlORION(
        image_queue=image_queue,
        user_message_queue=user_message_queue,
        bot_message_queue=bot_message_queue,
        chatgpt_config=AzureGPT4Config(),
        dump_dir="dump_dir",
        use_stream=True,
        record_interaction=False,
        use_memory=True,
        use_vlmap=True,
        fast_explore=True,
        display_shortside=480,
        save_dir_name="predict",
        scene_ids=["4ok3usBNeis"],
        floor_set=(-1, 1),
        auto_record=False,
        display_setting="rgb+topdownmap",
        display_horizontally=False,
        headless=True,
        use_gt_pose=True,
        load_existing_occumap=True,
        save_new_occumap=False,
    )

    game.run()

    p.join()


if __name__ == "__main__":
    main()
