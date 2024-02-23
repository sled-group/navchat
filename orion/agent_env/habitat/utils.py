import habitat
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import append_text_to_image, images_to_video

cv2 = try_cv2_import()


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def quiet():
    import os

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"


def update_fov(config, fov=90):
    config.defrost()
    config.SIMULATOR.DEPTH_SENSOR.HFOV = fov
    config.SIMULATOR.RGB_SENSOR.HFOV = fov
    config.SIMULATOR.SEMANTIC_SENSOR.HFOV = fov
    config.freeze()


def update_scene(config, split="val", scene_ids=["4ok3usBNeis"]):
    config.defrost()
    if split is not None:
        config.DATASET.SPLIT = split
    if scene_ids is not None:
        config.DATASET.CONTENT_SCENES = scene_ids
    config.freeze()


def update_holonomic_action(config):
    config.defrost()
    config.TASK.ACTIONS.MOVE_BACKWARD = habitat.config.Config()
    config.TASK.ACTIONS.MOVE_BACKWARD.TYPE = "MoveBackwardAction"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "Holonomic"
    config.freeze()


def add_top_down_map_and_collision(config):
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.MEASUREMENTS.append("COLLISIONS")
    config.freeze()


def try_action_import():
    try:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        MOVE_FORWARD = HabitatSimActions.MOVE_FORWARD
        TURN_LEFT = HabitatSimActions.TURN_LEFT
        TURN_RIGHT = HabitatSimActions.TURN_RIGHT
        STOP = HabitatSimActions.STOP
    except:
        MOVE_FORWARD = 0
        TURN_LEFT = 1
        TURN_RIGHT = 2
        STOP = 3
    return MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP


def try_action_import_v2():
    try:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        MOVE_FORWARD = HabitatSimActions.MOVE_FORWARD
        MOVE_BACKWARD = "MOVE_BACKWARD"
        TURN_LEFT = HabitatSimActions.TURN_LEFT
        TURN_RIGHT = HabitatSimActions.TURN_RIGHT
        # LOOK_UP = HabitatSimActions.LOOK_UP
        # LOOK_DOWN = HabitatSimActions.LOOK_DOWN
        STOP = HabitatSimActions.STOP
        return MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP
    except:
        raise NotImplementedError
