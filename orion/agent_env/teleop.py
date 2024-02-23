import cv2

from orion.agent_env.habitat.base import HabitatAgentEnv
from orion.agent_env.habitat.utils import try_action_import_v2

FORWARD_KEY = "w"
BACKWARD_KEY = "s"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "p"

MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP = try_action_import_v2()


class TeleOpAgentEnv(HabitatAgentEnv):
    def loop(self):
        while True:
            keystroke = cv2.waitKey(0)
            if keystroke == ord(FORWARD_KEY):
                action = MOVE_FORWARD
            elif keystroke == ord(BACKWARD_KEY):
                action = MOVE_BACKWARD
            elif keystroke == ord(LEFT_KEY):
                action = TURN_LEFT
            elif keystroke == ord(RIGHT_KEY):
                action = TURN_RIGHT
            elif keystroke == ord(FINISH):
                action = STOP
            else:
                print("INVALID KEY")
                action = None
            if action is not None:
                self.step(action)
