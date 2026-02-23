from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import logging
from gymnasium.spaces import Box, Discrete
from gymnasium import Wrapper

# logging filter for Opponent_agent
_original_get_logger = logging.getLogger

def _patched_get_logger(name=None):
    logger = _original_get_logger(name)
    if name and "Opponent_bot" in str(name):
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    return logger

logging.getLogger = _patched_get_logger

# SB3
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# SB3 MaskablePPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.battle import AbstractBattle, Battle
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import DefaultBattleOrder, RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration

from tqdm import tqdm

logging.getLogger("poke_env.player").setLevel(logging.WARNING)
logging.getLogger("agent").setLevel(logging.CRITICAL)


# -----------------------------
# Directories and constants
# -----------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

OBS_SIZE = 44
ACTION_SPACE_SIZE = 11

def model_path(name) -> Path:
    return MODEL_DIR / f"{name}.zip"


# -----------------------------
# Progress Callback
# -----------------------------
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.episode_count = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps", dynamic_ncols=True)
        print(f"[Training] Total timesteps this run: {self.total_timesteps:,} | Lifetime timesteps: {self.model._total_timesteps:,}")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        dones = self.locals.get("dones", [])
        if any(dones):
            self.episode_count += sum(dones)
            self.pbar.set_postfix(battles=self.episode_count, refresh=False)
        return True

    def _on_training_end(self):
        self.pbar.close()


# -----------------------------
# Custom Environment
# -----------------------------
class CustomEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: Discrete(ACTION_SPACE_SIZE)
            for agent in self.possible_agents
        }

        self.gen_data = GenData.from_gen(1)
        self.species_to_id = {
            name: i for i, name in enumerate(self.gen_data.pokedex.keys())
        }

        # Track previous battle state for step rewards
        self._prev_my_hp: Dict[str, float] = {}
        self._prev_opp_hp: Dict[str, float] = {}
        self._prev_opp_fainted: int = 0
        self._prev_my_fainted: int = 0

    def action_to_order(self, action, battle, fake=False, strict=True):
        if action == 10:
            action = -2
        return super().action_to_order(action, battle, fake=fake, strict=strict)

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        agent_config = AccountConfiguration("agent", None)
        opponent_config = AccountConfiguration("Opponent_bot", None)

        env = cls(
            battle_format=config["battle_format"],
            log_level=30,
            open_timeout=None,
            strict=False,
            account_configuration1=agent_config,
            account_configuration2=opponent_config
        )
        opponent = SimpleHeuristicsPlayer(start_listening=False, account_configuration=opponent_config)
        base_env = SingleAgentWrapper(env, opponent)
        return ActionMasker(base_env, mask_env)

    def _reset_battle_tracking(self):
        self._prev_my_hp = {}
        self._prev_opp_hp = {}
        self._prev_opp_fainted = 0
        self._prev_my_fainted = 0

    def calc_reward(self, battle) -> float:
        reward = 0.0

        # --- Win/loss/tie at battle end ---
        if battle.finished:
            if battle.won is None:
                reward += -0.5  # tie
            elif battle.won:
                reward += 1.0
            else:
                reward -= 1.0

            # HP differential bonus at end
            my_hp = sum(p.current_hp_fraction for p in battle.team.values())
            opp_hp = sum(p.current_hp_fraction for p in battle.opponent_team.values())
            reward += 0.25 * (my_hp - opp_hp)

            # Fainted penalties/bonuses at end
            fainted_own = sum(1 for p in battle.team.values() if p.fainted)
            fainted_enemy = sum(1 for p in battle.opponent_team.values() if p.fainted)
            reward -= 0.05 * fainted_own
            reward += 0.06 * fainted_enemy

            self._reset_battle_tracking()
            return reward

        # --- Step rewards during battle ---
        # Reward for new enemy faints this step
        opp_fainted_now = sum(1 for p in battle.opponent_team.values() if p.fainted)
        new_opp_faints = opp_fainted_now - self._prev_opp_fainted
        if new_opp_faints > 0:
            reward += 0.06 * new_opp_faints
        self._prev_opp_fainted = opp_fainted_now

        # Penalty for new own faints this step
        my_fainted_now = sum(1 for p in battle.team.values() if p.fainted)
        new_my_faints = my_fainted_now - self._prev_my_fainted
        if new_my_faints > 0:
            reward -= 0.05 * new_my_faints
        self._prev_my_fainted = my_fainted_now

        # Type effectiveness bonus for current move
        if battle.active_pokemon and battle.opponent_active_pokemon:
            for move in battle.available_moves:
                eff = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=self.gen_data.type_chart
                )
                if eff > 1:
                    reward += 0.02
                elif eff < 1:
                    reward -= 0.01

                # STAB bonus
                if battle.active_pokemon.type_1 == move.type or battle.active_pokemon.type_2 == move.type:
                    reward += 0.02

        return reward

    def embed_battle(self, battle: AbstractBattle):
        try:
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
            special_case = np.zeros(2, dtype=np.float32)

            for i, (_, mon) in enumerate(sorted(battle.team.items())):
                team_hp_ratio[i] = -1.0 if mon.fainted or mon.max_hp == 0 else (mon.current_hp / mon.max_hp) * 2 - 1

            for i, (_, mon) in enumerate(sorted(battle.opponent_team.items())):
                opponent_hp_ratio[i] = -1.0 if mon.fainted or mon.max_hp == 0 else (mon.current_hp / mon.max_hp) * 2 - 1

            for i, move in enumerate(battle.available_moves):
                try:
                    moves_base_power[i] = (move.base_power / 250) * 2 - 1 if move.base_power is not None else 0.0
                    if battle.opponent_active_pokemon is not None:
                        moves_dmg_multiplier[i] = np.clip(
                            move.type.damage_multiplier(
                                battle.opponent_active_pokemon.type_1,
                                battle.opponent_active_pokemon.type_2,
                                type_chart=self.gen_data.type_chart
                            ) - 1.0, -1.0, 1.0
                        )
                    moves_pp_ratio[i] = (move.current_pp / move.max_pp) * 2 - 1
                except AssertionError:
                    pass

            for i, mon in enumerate(battle.team.values()):
                if mon.fainted:
                    self_team_status[i] = -1.0
            for i, mon in enumerate(battle.opponent_team.values()):
                if mon.fainted:
                    opponent_team_status[i] = -1.0

            species_list = list(self.gen_data.pokedex.keys())
            for i, (_, mon) in enumerate(sorted(battle.team.items())):
                if mon is not None and mon.species is not None:
                    idx = species_list.index(mon.species.lower()) if mon.species.lower() in species_list else 0
                    team_identifier[i] = (idx / (len(species_list) - 1)) * 2 - 1

            if len(battle.available_moves) == 0 and len(battle.available_switches) == 0:
                special_case[0] = 1
            if battle.active_pokemon.must_recharge:
                special_case[1] = 1

            return np.float32(np.concatenate([
                moves_base_power,
                moves_dmg_multiplier,
                moves_pp_ratio,
                self_team_status,
                opponent_team_status,
                team_hp_ratio,
                opponent_hp_ratio,
                team_identifier,
                special_case,
            ]))

        except AssertionError:
            return np.zeros(OBS_SIZE, dtype=np.float32)

    def reset(self, *args, **kwargs):
        self._reset_battle_tracking()
        return super().reset(*args, **kwargs)

    def close(self):
        super().close()
        print("[Environment Closed]")

    def step(self, action):
        return super().step(action)


# -----------------------------
# Masking function
# -----------------------------
def mask_env(env):
    battle = env.env.battle1
    action_mask = np.zeros(env.action_space.n, dtype=np.int8)

    if battle is None or battle.active_pokemon is None:
        return action_mask

    team = list(battle.team.values())
    available_switches = set(battle.available_switches)
    moves = list(battle.active_pokemon.moves.values())
    available_moves = set(battle.available_moves)
    move_offset = 6
    choose_default = 10

    if len(available_moves) == 0 and len(available_switches) == 0:
        action_mask[choose_default] = 1
        return action_mask

    if battle.active_pokemon.must_recharge:
        action_mask[choose_default] = 1
        return action_mask

    if not battle.force_switch or not battle.active_pokemon.fainted:
        for slot, move in enumerate(moves):
            if move in available_moves:
                action_mask[slot + move_offset] = 1

    for slot, mon in enumerate(team):
        if mon in available_switches:
            action_mask[slot] = 1

    return action_mask


# -----------------------------
# Training functions
# -----------------------------
def make_train_env():
    return CustomEnv.create_single_agent_env({"battle_format": "gen1randombattle"})


def train_new(model_name, timesteps):
    path = model_path(model_name)
    if path.exists():
        print(f"Model '{model_name}' already exists! Delete it or choose another name.")
        return

    print(f"[Training Started] model={model_name}, timesteps={timesteps:,}")
    train_env = make_train_env()
    model = MaskablePPO("MlpPolicy", train_env, verbose=0, tensorboard_log="./tensorboard_logs/")
    model.learn(total_timesteps=timesteps, callback=ProgressCallback(timesteps), tb_log_name=model_name)
    model.save(path)

    train_env.close()
    print(f"[Model Saved] {model_name}")


def train_continue(model_name, timesteps):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found! Train a new model first.")
        return

    print(f"[Continuing Training] model={model_name}, additional timesteps={timesteps:,}")
    train_env = make_train_env()
    model = MaskablePPO.load(path, env=train_env, tensorboard_log="./tensorboard_logs/")
    model.learn(total_timesteps=timesteps, callback=ProgressCallback(timesteps), reset_num_timesteps=False, tb_log_name=model_name)
    model.save(path)

    train_env.close()
    print(f"[Model Saved] {model_name}")


def eval_model(model_name, n_battles=100):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found!")
        return

    print(f"[Evaluating] model={model_name}, battles={n_battles}")
    eval_env = make_train_env()
    model = MaskablePPO.load(path, env=eval_env)

    wins, losses, draws = 0, 0, 0
    pbar = tqdm(total=n_battles, desc="Evaluating", unit="battles")

    obs, _ = eval_env.reset()
    battles_done = 0

    while battles_done < n_battles:
        action_masks = get_action_masks(eval_env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        try:
            obs, reward, terminated, truncated, info = eval_env.step(action)
        except AssertionError:
            obs, _ = eval_env.reset()
            continue

        if terminated or truncated:
            battle = eval_env.env.env.battle1
            if battle is not None and battle.won is True:
                wins += 1
            elif battle is not None and battle.won is False:
                losses += 1
            else:
                draws += 1
            battles_done += 1
            pbar.update(1)
            pbar.set_postfix(W=wins, L=losses, D=draws, WR=f"{wins/battles_done:.1%}")
            obs, _ = eval_env.reset()

    pbar.close()
    eval_env.close()

    print(f"\n[Results] Battles: {n_battles} | Wins: {wins} | Losses: {losses} | Draws: {draws} | Win Rate: {wins/n_battles:.1%}")
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / n_battles}


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    MODE = "new"        # "new" | "continue" | "eval"
    MODEL_NAME = "RewardTest"
    training_steps = 100000

    if MODE == "new":
        train_new(MODEL_NAME, training_steps)
    elif MODE == "continue":
        train_continue(MODEL_NAME, training_steps)
    elif MODE == "eval":
        eval_model(MODEL_NAME, n_battles=100)

# For tensorboard run command in separate terminal:      tensorboard --logdir ./tensorboard_logs/