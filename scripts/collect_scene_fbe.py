import argparse
import time
from orion.agent_env.fbe import FBEAgentEnv
from orion.config.my_config import SCENE_ID_FLOOR_SET


def collect_data_one_scene(scene_id, floor):
    game = FBEAgentEnv(
        scene_ids=[scene_id],
        floor_set=floor,
        fast_explore=False,
        display_shortside=256,
        save_dir_name="predict",
        auto_record=True,
        display_setting="rgb+occumap+topdownmap",
        headless=True,
        use_gt_pose=True,
        load_existing_occumap=False,
        save_new_occumap=True,
    )
    game.run()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--scene_id",
        type=str,
        default="4ok3usBNeis",
        help="scene id, either 'all' or a specific scene id in SCENE_ID_FLOOR_SET",
    )
    args = argparser.parse_args()

    scene_dic = {item[0]: item for item in SCENE_ID_FLOOR_SET}
    if args.scene_id == "all":
        for scene_id, floor in SCENE_ID_FLOOR_SET:
            collect_data_one_scene(scene_id, floor)
            time.sleep(5)
    else:
        assert args.scene_id in scene_dic
        collect_data_one_scene(scene_dic[args.scene_id][0], scene_dic[args.scene_id][1])
