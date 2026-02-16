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


# embed_battle return vector size (must match the actual return size)
OBS_SIZE = 42

# Custom Environment
class CustomEnv(SinglesEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
        agent: Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )
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

    # Rewarding function (calculates per turn) - need to create a reward system
    def calc_reward(self, battle) -> float:
        reward = self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0)
        #print (reward)
        #print ("------------------------------------------------------------------")
        return reward


    # Observer function [-1, 1]
    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        moves = 4
        pokemon_team = 6
        moves_base_power = -np.ones(moves)
        moves_dmg_multiplier = np.ones(moves)
        moves_pp_ratio = np.zeros(moves)
        team_hp_ratio = np.ones(pokemon_team)
        opponent_hp_ratio = np.ones(pokemon_team)
        opponent_team_status = np.ones(pokemon_team, dtype=np.float32)
        self_team_status = np.ones(pokemon_team, dtype=np.float32)
        team_identifier = np.zeros(pokemon_team, dtype=np.float32)


        # hp percentages of both my team and the enemy
        for i, (_, mon) in enumerate(sorted(battle.team.items())):
            if mon.fainted or mon.max_hp == 0:
                team_hp_ratio[i] = -1.0
            else:
                team_hp_ratio[i] = (mon.current_hp / mon.max_hp) * 2 - 1

        for i, (_, mon) in enumerate(sorted(battle.opponent_team.items())):
            if mon.fainted or mon.max_hp == 0:
                opponent_hp_ratio[i] = -1.0
            else:
                opponent_hp_ratio[i] = (mon.current_hp / mon.max_hp) * 2 - 1

        # moves
        for i, move in enumerate(battle.available_moves):
            # move base_power
            if move.base_power is not None:
                moves_base_power[i] = (move.base_power / 250) * 2 - 1
            else:
                moves_base_power[i] = 0.0

            # move damage multiplier
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = np.clip(
                    move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=GenData.from_gen(battle.gen).type_chart
                    ) - 1.0, -1.0, 1.0) # Normalized to [-1, 1]

            # move pp
            moves_pp_ratio[i] = (move.current_pp / move.max_pp) * 2 - 1

        # Encoded team composition (1 = alive, -1 = fainted)
        for i, mon in enumerate(battle.team.values()):
            if mon.fainted:
                self_team_status[i] = -1.0

        # Encoded enemy team composition (1 = alive, -1 = fainted)
        for i, mon in enumerate(battle.opponent_team.values()):
            if mon.fainted:
                opponent_team_status[i] = -1.0 

        # Pokemon team - pokemon team's identifiers
        pokedex = GenData.from_gen(battle.gen).pokedex
        species_list = list(pokedex.keys())

        for i, (_, mon) in enumerate(sorted(battle.team.items())):
            if mon is not None and mon.species is not None:
                mons_species_id = mon.species.lower() # species to lower‑case
                if mons_species_id in species_list:
                    idx = species_list.index(mons_species_id)
                    team_identifier[i] = (idx / (len(species_list) - 1)) * 2 - 1 
                else:
                    team_identifier[i] = 0.0  # unknown
            else:
                team_identifier[i] = -1  # no mon

        # Final vector with 42 components
        final_vector = np.concatenate(
            [
                moves_base_power, # 4
                moves_dmg_multiplier, # 4
                moves_pp_ratio, # 4
                self_team_status, # 6
                opponent_team_status, # 6
                team_hp_ratio, # 6
                opponent_hp_ratio, # 6
                team_identifier, # 6
            ]
        )
        # observation vector
        return np.float32(final_vector)
    

    # reset function to make sure env resets after each battle
    def reset(self, *args, **kwargs):
        obs, infos = super().reset(*args, **kwargs)
        print("Environment reset! New battle started.")
        return obs, infos

    # close function to make sure the env closes after each interval
    def close(self):
        super().close()
        print("Environment closed.")
            # add proper closing to the final battle


# masking function
def mask_env(env):
    # unwrap ActionMasker -> SingleAgentWrapper -> CustomEnv
    battle = env.env.battle1
    action_mask = np.zeros(env.action_space.n, dtype=np.int8)

    if battle is None or battle.active_pokemon is None:
        return action_mask

    # forced states 
    # - recharge replaces the move that needs recharging ex: [1,2,Hyper_beam,4] -> [1,2,recharge,4]. problem because the initial list doesn't account for recharge
    # - [/choosing move 1] being the only valid order (idk what it pertains to or what conditions need to be met) - maybe dig or encore




    # available moves
    move_offset = 6 
    available = set(battle.available_moves)
    moves = list(battle.active_pokemon.moves.values())

    for slot, move in enumerate(moves):
        if move in available:
            action_mask[slot + move_offset] = 1

    # available switches 
    team = list(battle.team.values())
    available_switches = set(battle.available_switches)

    for slot, mon in enumerate(team):
        if mon in available_switches:
            action_mask[slot] = 1
    
   
    #print (action_mask)
    #print (team)
    #print (moves)
    #print (battle.available_moves)
    #print (battle.available_switches)
    #print ("--------------------------------------------------------------------------------")

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
    training_steps = 2000

    if MODE == "new":
        train_new(MODEL_NAME, training_steps)
