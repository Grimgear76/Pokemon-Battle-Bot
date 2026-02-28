from typing import Any, Dict, Optional
from pathlib import Path
import logging

import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.battle import AbstractBattle, Battle
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import ForfeitBattleOrder, _EmptyBattleOrder
from poke_env import AccountConfiguration


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def model_path(name) -> Path:
    return MODEL_DIR / f"{name}.zip"


OBS_SIZE = 42


class AutoResetWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._dummy_obs = np.zeros(OBS_SIZE, dtype=np.float32)

    def _get_inner_env(self):
        try:
            return self.env.env.env
        except AttributeError:
            return self.env.env

    def _force_finish_all(self):
        try:
            inner = self._get_inner_env()
            for attr in ("agent1", "agent2"):
                agent = getattr(inner, attr, None)
                if agent:
                    for b in agent._battles.values():
                        b._finished = True
        except Exception:
            pass

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except (AssertionError, RuntimeError):
            inner = self._get_inner_env()
            try:
                for agent_attr, battle_attr in (("agent1", "battle1"), ("agent2", "battle2")):
                    agent = getattr(inner, agent_attr)
                    unfinished = [b for b in agent._battles.values() if not b.finished]
                    if unfinished:
                        setattr(inner, battle_attr, unfinished[0])
                        agent.battle = unfinished[0]
                inner.agent1_to_move = True
                inner.agent2_to_move = True
            except Exception as e:
                print(f"Cleanup error: {e}")
            obs = self._dummy_obs
            reward = 0.0
            terminated = True
            truncated = False
            info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        inner = self._get_inner_env()
        for attr in ("agent1", "agent2"):
            agent = getattr(inner, attr, None)
            if agent and not agent.battle_queue.empty():
                try:
                    agent.battle_queue.get()
                except Exception:
                    pass
        return self.env.reset(**kwargs)

    def close(self):
        self._force_finish_all()
        try:
            self.env.close()
        except Exception:
            pass


class CustomEnv(SinglesEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        agent_config = AccountConfiguration("agent", None)
        opponent_config = AccountConfiguration("random_bot", None)

        env = cls(
            battle_format=config["battle_format"],
            log_level=25,
            open_timeout=None,
            strict=False,
            account_configuration1=agent_config,
            account_configuration2=opponent_config,
        )

        opponent = SimpleHeuristicsPlayer(
            start_listening=False,
            account_configuration=opponent_config,
        )

        logging.getLogger("random_bot").setLevel(logging.ERROR)

        base_env = SingleAgentWrapper(env, opponent)
        masked_env = ActionMasker(base_env, mask_env)
        return AutoResetWrapper(masked_env)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

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

        for i, move in enumerate(battle.available_moves):
            if move.base_power is not None:
                moves_base_power[i] = (move.base_power / 250) * 2 - 1
            else:
                moves_base_power[i] = 0.0

            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = np.clip(
                    move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=GenData.from_gen(battle.gen).type_chart,
                    ) - 1.0, -1.0, 1.0,
                )

            moves_pp_ratio[i] = (move.current_pp / move.max_pp) * 2 - 1

        for i, mon in enumerate(battle.team.values()):
            if mon.fainted:
                self_team_status[i] = -1.0

        for i, mon in enumerate(battle.opponent_team.values()):
            if mon.fainted:
                opponent_team_status[i] = -1.0

        pokedex = GenData.from_gen(battle.gen).pokedex
        species_list = list(pokedex.keys())

        for i, (_, mon) in enumerate(sorted(battle.team.items())):
            if mon is not None and mon.species is not None:
                mons_species_id = mon.species.lower()
                if mons_species_id in species_list:
                    idx = species_list.index(mons_species_id)
                    team_identifier[i] = (idx / (len(species_list) - 1)) * 2 - 1
                else:
                    team_identifier[i] = 0.0
            else:
                team_identifier[i] = -1

        return np.float32(np.concatenate([
            moves_base_power,       # 4
            moves_dmg_multiplier,   # 4
            moves_pp_ratio,         # 4
            self_team_status,       # 6
            opponent_team_status,   # 6
            team_hp_ratio,          # 6
            opponent_hp_ratio,      # 6
            team_identifier,        # 6
        ]))

    def reset(self, *args, **kwargs):
        obs, infos = super().reset(*args, **kwargs)
        print("Environment reset! New battle started.")
        return obs, infos

    def close(self):
        super().close()
        print("Environment closed.")


def mask_env(env):
    try:
        battle = env.env.env.battle1
    except AttributeError:
        battle = env.env.battle1

    action_mask = np.zeros(env.action_space.n, dtype=np.int8)
    move_offset = 6

    if battle is None or battle.active_pokemon is None or battle.finished:
        action_mask[move_offset] = 1
        return action_mask

    available_moves = battle.available_moves
    available_switches = battle.available_switches

    if battle.force_switch:
        team = list(battle.team.values())
        for slot, mon in enumerate(team):
            if mon in set(available_switches):
                action_mask[slot] = 1

    elif len(available_moves) == 1:
        # Forced single move: recharge, sky attack turn 2, solar beam, etc.
        action_mask[move_offset] = 1

    else:
        for i in range(len(available_moves)):
            action_mask[move_offset + i] = 1
        if not battle.trapped:
            team = list(battle.team.values())
            for slot, mon in enumerate(team):
                if mon in set(available_switches):
                    action_mask[slot] = 1

    if action_mask.sum() == 0:
        action_mask[move_offset] = 1

    return action_mask


def make_train_env():
    return CustomEnv.create_single_agent_env({"battle_format": "gen5randombattle"}) #Choose Generation


def train_new(model_name, timesteps):
    path = model_path(model_name)
    if path.exists():
        print(f"Model '{model_name}' already exists! Choose another name or use MODE = 'continue'.")
        return

    print(f"Starting new training run: {model_name}")
    train_env = make_train_env()
    model = MaskablePPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(path)
    train_env.close()
    print(f"Saved new model: {path}")


def train_continue(model_name, timesteps):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found at {path}. Train a new model first with MODE = 'new'.")
        return

    print(f"Continuing training from: {path}")
    train_env = make_train_env()
    model = MaskablePPO.load(path, env=train_env)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    model.save(path)
    train_env.close()
    print(f"Saved continued model: {path}")


if __name__ == "__main__":
    MODE = "new"        # new | continue     # Change this to continue with previous training data
    MODEL_NAME = "testTrash"
    TRAINING_STEPS = 5000

    if MODE == "new":
        train_new(MODEL_NAME, TRAINING_STEPS)
    elif MODE == "continue":
        train_continue(MODEL_NAME, TRAINING_STEPS)
    else:
        print(f"Unknown MODE: '{MODE}'. Use 'new' or 'continue'.")
