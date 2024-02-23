"""given the recognition probability, find out the true postive images"""

import numpy as np

from orion import logger

np.set_printoptions(precision=3, suppress=True, linewidth=200)


class CLIPScorer:
    def __init__(self):
        self.prob_list = []
        self.masktime = 0
        self.found_goal = False

    def reset(self):
        self.prob_list = []
        self.masktime = 0

    def add_prob(self, prob):
        self.prob_list.append(prob)

    def is_goal_found(self, prob):
        if len(self.prob_list) > 0:
            mean_prob = np.mean(self.prob_list)
        else:
            mean_prob = 0
        self.add_prob(prob)
        if mean_prob > 0.9:
            theshold = 0.99
        elif mean_prob > 0.8:
            theshold = 0.95
        elif mean_prob > 0.6:
            theshold = 0.9
        else:
            theshold = 0.8

        # logger.info(
        #     f"\t mean prob {mean_prob:2f}, theshold {theshold}, prob {prob:3f}, {self.prob_list[-3:]}, {np.mean(self.prob_list[-3:])}"
        # )
        if (
            prob > theshold
            and len(self.prob_list) > 3
            and np.mean(self.prob_list[-3:]) > theshold
            and self.masktime == 0
        ):
            logger.info("\033[31m [CLIP] detect a pulse\033[m")
            self.found_goal = True
            return True
        elif (
            len(self.prob_list) > 1
            and prob - max(self.prob_list[-2], mean_prob) > 0.5
            and self.masktime == 0
        ):
            logger.info("\033[31m [CLIP] detect a pulse\033[m")
            self.found_goal = True
            return True
        elif (
            len(self.prob_list) > 3
            and self.prob_list[-1] - max(self.prob_list[-3], mean_prob) > 0.5
            and self.prob_list[-2] - max(self.prob_list[-3], mean_prob) > 0.4
            and self.masktime == 0
        ):
            logger.info("\033[31m [CLIP] detect a pulse\033[m")
            self.found_goal = True
            return True
        else:
            self.masktime = max(0, self.masktime - 1)
            self.found_goal = False
            return False

    def set_masktime(self, masktime=5):
        logger.info(f" CLIP set masktime to {masktime}")
        self.masktime = masktime
