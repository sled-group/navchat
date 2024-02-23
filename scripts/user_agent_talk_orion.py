import json
import os
import re
import time
from orion import logger
from orion.agent_env.chatgpt_control_orion import ChatGPTControlORION
from orion.user_simulator.chatgpt_based_sim import (
    ChatGPTUserSimulator,
    CountourMaskPrediction,
)
from orion.abstract.interaction_history import SucMsg
from orion.config.chatgpt_config import *


class ChatGPTControlAndUserSim(ChatGPTControlORION):
    def __init__(self, max_trial, max_round, category, chatgpt_usrsim_config, clear_gptctx=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.usr_sim = ChatGPTUserSimulator(
            chatgpt_usrsim_config=chatgpt_usrsim_config,
            scene_id=kwargs["scene_ids"][0],
            floor_plan=kwargs["floor_set"],
            max_trial=max_trial,
            max_round=max_round,
            category=category,
            is_cow_baseline=self.is_cow_baseline,
            is_vlamp_baseline=self.is_vlmap_baseline,
        )
        logger.info("User Simulator Initialized.")
        logger.info(f"Max Trial: {max_trial}, Max Round: {max_round}, Cat: {category}")
        for k, v in self.usr_sim.topo_graph.instance_dict.items():
            logger.info(f"Instance {k}: {v}")

        logger.info("User Goal init")
        for g in self.usr_sim.goal_gen.goals:
            logger.info(g)

        self.clear_gptctx = clear_gptctx
        assert self.is_vlmap_baseline is False
        assert self.is_cow_baseline is False

        suffix = f"orion_t{max_trial}r{max_round}_{category}"
        if self.use_memory:
            suffix += "_mem"
        else:
            suffix += "_nomem"
        if self.use_vlmap:
            suffix += "_vmp"
        else:
            suffix += "_novmp"
        if self.use_explore:
            suffix += "_exp"
        else:
            suffix += "_noexp"
        if self.clear_gptctx:
            suffix += "_noctx"  # clear every new round
        else:
            suffix += "_ctx"

        logger.info(f"Dump dir suffix: {suffix}")
        self.dump_dir = os.path.join(self.save_dir, f"dump_{suffix}")
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)
        else:
            logger.warning(f"Dump dir {self.dump_dir} already exists!")
            input("Press Enter to continue...")
        self.is_first_turn = True

    def _get_user_input(self):
        logger.info("\nGenreate User Utterance with GPT Simulator...")
        is_first_turn = self.is_first_turn
        if self.is_first_turn:
            agtresponse = "Hello, what should I do?"
            self.is_first_turn = False
        else:
            agtresponse = self.agent_response

        agt_predict = CountourMaskPrediction(
            predict_contours=self.predict_contours,
            predict_masks=self.predict_masks,
        )

        task_finished, is_new_goal, is_new_round, goal_succ, instruction = self.usr_sim.step(
            agent_response=agtresponse,
            agtpose=self.agent_state.pose_2d,
            semantic_img=self.observations.semantic,
            agt_predict=agt_predict,
            step_count=self.step_count,
            first_turn=is_first_turn,
        )
        self.task_finished = task_finished
        logger.info(f"[User Simulator] {instruction}")
        self.interaction_history.append(
            SucMsg(reward=goal_succ)
        )
        if is_new_round:
            self.usr_sim.goal_gen._is_new_round = False
            last_round = self.usr_sim.goal_gen.last_round -1
            logger.info(f"Start new Round! From round {last_round} to {last_round+1}")
            if self.use_memory:
                self.object_memory.save(self.dump_dir, suffix=f"_round{last_round}")
            
            gpt_context = self.chatgpt.messages
            gpt_context_path = os.path.join(self.dump_dir, f"gptctx_round{last_round}.json")
            json.dump(gpt_context, open(gpt_context_path, "w"))
            
            if self.clear_gptctx: 
                self.chatgpt.clear_ctx()
            
        if is_new_goal:
            # save early
            eval_result_path = os.path.join(self.dump_dir, "result.json")
            self.usr_sim.goal_gen.save(eval_result_path)
            conversation_path = os.path.join(self.dump_dir, "dialog.json")
            json.dump(self.record_conversations, open(conversation_path, "w"))
            
            # save money
            if re.search(r"(gpt-4|gpt4)", self.chatgpt.model):
                self.chatgpt.clear_ctx()
        
            
        return instruction

    def save(self):
        super().save()
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        self.interaction_history.save(self.dump_dir)
        if self.use_memory:
            self.object_memory.save(self.dump_dir, suffix="_final")        
        eval_result_path = os.path.join(self.dump_dir, "result.json")
        results = self.usr_sim.goal_gen.save(eval_result_path)
        logger.info(f"Dumped user simulator result to {eval_result_path}")
        conversation_path = os.path.join(self.dump_dir, "dialog.json")
        json.dump(self.record_conversations, open(conversation_path, "w"))
        logger.info(f"Dumped conversation to {conversation_path}")

        try:
            self.usr_sim.eval(results)
        except:
            pass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, default="4ok3usBNeis")
    parser.add_argument("--floor_b", type=int, default=-1)
    parser.add_argument("--floor_u", type=int, default=1)
    parser.add_argument("--category", type=str, default="mixed", choices=["landmark", "instruction", "description", "correction", "mixed", "none"])
    args = parser.parse_args()

    max_trial=5
    max_round=1
    category=args.category
    use_memory=True
    use_vlmap=True
    use_explore=True
    clear_gptctx=False

    chatgpt_config=AzureGPT4Config()
    chatgpt_usrsim_config=AzureGPT35Config()

    game = ChatGPTControlAndUserSim(
        max_trial=max_trial,
        max_round=max_round,
        category=category,
        chatgpt_config=chatgpt_config,
        chatgpt_usrsim_config=chatgpt_usrsim_config,
        use_memory=use_memory,
        use_vlmap=use_vlmap,
        use_explore=use_explore,
        clear_gptctx=clear_gptctx,
        record_interaction=True,
        use_stream=False,
        fast_explore=True,
        scene_ids=[args.scene_id],
        floor_set=(args.floor_b, args.floor_u),
        display_shortside=256,
        save_dir_name="predict",
        auto_record=False,
        display_setting="rgb+occumap+topdownmap",
        headless=True,
        use_gt_pose=True,
        load_existing_occumap=True,
        save_new_occumap=False,
    )

    game.run()
