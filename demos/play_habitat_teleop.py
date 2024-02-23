from orion.agent_env.teleop import TeleOpAgentEnv


# pick a scene from orion.config.my_config.SCENE_ID_FLOOR_SET
game = TeleOpAgentEnv(
    scene_ids=["MHPLjHsuG27"],
    floor_set=(-2, 2),
    display_shortside=256,
    save_dir_name="teleop",
    auto_record=False,
    display_setting="rgb+topdownmap",
    use_gt_pose=True,
    load_existing_occumap=True,
)
game.run()
