from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, Space

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from poke_env.battle import AbstractBattle, Battle
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration


# Custom Environment with custom rewards, observations, 
class CustomEnv(SinglesEnv[npt.NDArray[np.float32]]):
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

        env = cls(battle_format=config["battle_format"], log_level=25, open_timeout=None, strict=False, account_configuration1=agent_config, account_configuration2=opponent_config)
        opponent = RandomPlayer(start_listening=False, account_configuration=opponent_config)
        return SingleAgentWrapper(env, opponent)


    # Rewarding function
    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0)


    # Observer function
    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)

        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (move.base_power / 100)

            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(battle.gen).type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = (len([mon for mon in battle.team.values() if mon.fainted]) / 6)
        fainted_mon_opponent = (len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6)

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

    # masking function (change)
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        return mask


if __name__ == "__main__":
    def make_env():
        return CustomEnv.create_single_agent_env({"battle_format": "gen1randombattle"})


    train_env = DummyVecEnv([make_env])

    # mode made with SB3 algorithm and our customEnv
    model = PPO("MlpPolicy", train_env, verbose=1)

    # training loop (total_timesteps is the number of turns the agent takes)
    model.learn(total_timesteps=4095)

    train_env.envs[0].env.close() # Force closes last battle and makes agent forfeit, Change it

    model.save("example_env_ppo")

    print("Training finished and model saved!")