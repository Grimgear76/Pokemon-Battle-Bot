from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, Space
from gymnasium import Wrapper

# SB3 PPO
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv

# SB3 MaskablePPO 
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.battle import AbstractBattle, Battle
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def model_path(name) -> Path:
    return MODEL_DIR / f"{name}.zip"


# Custom Environment with custom rewards, observations, 
class CustomEnv(SinglesEnv):
    LOW = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    HIGH = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(np.array(self.LOW, dtype=np.float32), np.array(self.HIGH, dtype=np.float32), dtype=np.float32)
            for agent in self.possible_agents
        }

    # create environment function (single agent)
    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        agent_config = AccountConfiguration("agent", None)
        opponent_config = AccountConfiguration("random_bot", None)

        # Class Constructor
        env = cls(battle_format=config["battle_format"], log_level=25, open_timeout=None, strict=False, account_configuration1=agent_config, account_configuration2=opponent_config)
        
        opponent = SimpleHeuristicsPlayer(start_listening=False, account_configuration=opponent_config)
        
        base_env = SingleAgentWrapper(env, opponent)
        return ActionMasker(base_env, mask_env)

    # Rewarding function
    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0)


    # Observer function
    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)

        # move power
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (move.base_power / 100)
            # move damage multiplier
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(battle.gen).type_chart
                )
        # move pp (add this feature)

        # We count how many pokemons have fainted in each team
        fainted_mon_team = (len([mon for mon in battle.team.values() if mon.fainted]) / 6)
        fainted_mon_opponent = (len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6)

        # Pokemon team (add this feature)

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        # observation vector
        return np.float32(final_vector)
    

# masking function (fixing masking)
def mask_env(env):
    # unwrap ActionMasker -> SingleAgentWrapper -> CustomEnv
    poke_env = env.env.battle1

    action_mask = np.zeros(env.action_space.n, dtype=np.int8)
    
    battle = poke_env
    if battle is None:
        return action_mask

    # available moves
    move_offset = 6 
    for i, move in enumerate(battle.available_moves):
        action_mask[i + move_offset] = 1

    
    for i in range(len(battle.available_switches)): 
        action_mask[i] = 1
    
    print (battle.available_moves)
    print (battle.available_switches)
    print ("--------------------------------------------------------------------------------")

    return action_mask


    


# -----------------------------------------------------
# Training and Evaluation functions
# -----------------------------------------------------

def make_train_env():
    return CustomEnv.create_single_agent_env({"battle_format": "gen1randombattle"})

def train_new(model_name, timesteps):
    path = model_path(model_name)
    if path.exists():
        print(f"Model {model_name} already exists! Choose another name.")
        return

    train_env = make_train_env()
    model = MaskablePPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(path)

    train_env.close()

    print(f"Saved new model: {model_name}")




if __name__ == "__main__":
    MODE = "new"   # "new" | "continue" | "eval"
    MODEL_NAME = "testTrash"
    training_steps = 6000

    if MODE == "new":
        train_new(MODEL_NAME, training_steps)
