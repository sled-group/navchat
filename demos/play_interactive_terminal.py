import argparse
from orion.agent_env.chatgpt_control_orion import ChatGPTControlORION
from orion.agent_env.chatgpt_control_cow import ChatGPTControlCoW
from orion.agent_env.chatgpt_control_vlmap import ChatGPTControlVLMap
from orion.agent_env.hybrid_search import HybridSearchAgentEnv
from orion.config.chatgpt_config import (
    AzureGPT35Config,
    AzureGPT4Config,
    OpenAIGPT35Config,
    OpenAIGPT4Config,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-type", choices=["openai", "azure"], default="azure")
    parser.add_argument("--model-type", choices=["gpt35", "gpt4"], default="gpt4")
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--use_memory", type=bool, default=True)
    parser.add_argument("--use_vlmap", type=bool, default=True)
    parser.add_argument("--fast_explore", action="store_true", default=False)
    parser.add_argument("--scene_id", type=str, default="4ok3usBNeis")
    parser.add_argument("--floor_set", type=int, nargs=2, default=(-1, 1))
    parser.add_argument("--use_chatgpt", type=bool, default=True)
    parser.add_argument(
        "--method-type", type=str, default="orion", choices=["orion", "vlmap", "cow"]
    )
    parser.add_argument(
        "--vlmap_dir",
        type=str,
        default="lseg_vlmap",
        choices=["lseg_vlmap", "conceptfusion_vlmap"],
    )
    parser.add_argument("--dump_dir", type=str, default="dump")
    parser.add_argument("--record_interaction", type=bool, default=False)

    args = parser.parse_args()

    if args.api_type == "openai":
        if args.model_type == "gpt35":
            chatgpt_config = OpenAIGPT35Config()
        elif args.model_type == "gpt4":
            chatgpt_config = OpenAIGPT4Config()  # type: ignore
        else:
            raise ValueError("model_type can only be ['gpt35', 'gpt4']")
    elif args.api_type == "azure":
        if args.model_type == "gpt35":
            chatgpt_config = AzureGPT35Config()
        elif args.model_type == "gpt4":
            chatgpt_config = AzureGPT4Config()
        else:
            raise ValueError("model_type can only be ['gpt35', 'gpt4']")

    if args.method_type == "orion":
        is_vlmap_baseline = False
        is_cow_baseline = False
        ChatGPTControl = ChatGPTControlORION
        dump_dir = args.dump_dir + "/orion"
    elif args.method_type == "vlmap":
        is_vlmap_baseline = True
        is_cow_baseline = False
        ChatGPTControl = ChatGPTControlVLMap
        dump_dir = args.dump_dir + "/vlmap"
    elif args.method_type == "cow":
        is_vlmap_baseline = False
        is_cow_baseline = True
        ChatGPTControl = ChatGPTControlCoW
        dump_dir = args.dump_dir + "/cow"
    else:
        raise ValueError("method_type can only be ['orion', 'vlmap', 'cow']")

    if args.use_chatgpt:
        # talk in natural language in the cmd line, e.g. "go to the shelf"
        game = ChatGPTControl(
            use_stream=args.stream,
            use_memory=args.use_memory,
            use_vlmap=args.use_vlmap,
            fast_explore=args.fast_explore,
            display_shortside=256,
            save_dir_name="predict",
            auto_record=False,
            display_setting="rgb+occumap+topdownmap",
            headless=False,
            use_gt_pose=True,
            load_existing_occumap=True,
            save_new_occumap=False,
            scene_ids=[args.scene_id],
            floor_set=args.floor_set,
            chatgpt_config=chatgpt_config,
            vlmap_dir=args.vlmap_dir,
            is_vlmap_baseline=is_vlmap_baseline,
            is_cow_baseline=is_cow_baseline,
            dump_dir=dump_dir,
            record_interaction=args.record_interaction,
        )

    else:
        # talk with restricted inputs
        # the input string is (phrase|noun), e.g. "red apple|apple"
        # or just input a noun, e.g. "apple" to the cmd line
        game = HybridSearchAgentEnv(
            use_memory=args.use_memory,
            use_vlmap=args.use_vlmap,
            fast_explore=args.fast_explore,
            scene_ids=[args.scene_id],
            floor_set=args.floor_set,
            display_shortside=256,
            save_dir_name="predict",
            auto_record=False,
            display_setting="rgb+occumap+topdownmap",
            use_gt_pose=True,
            load_existing_occumap=True,
            save_new_occumap=False,
            vlmap_dir=args.vlmap_dir,
            is_vlmap_baseline=is_vlmap_baseline,
            is_cow_baseline=is_cow_baseline,
        )

    game.run()
