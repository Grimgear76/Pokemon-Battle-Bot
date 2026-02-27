from typing import Any, Dict, Optional
from pathlib import Path
import multiprocessing
import asyncio

import numpy as np
import torch
import torch.nn as nn
import logging
from gymnasium.spaces import Box, Discrete
from gymnasium import Wrapper

# logging filter for Opponent_agent and noisy agent loggers
_original_get_logger = logging.getLogger

def _patched_get_logger(name=None):
    logger = _original_get_logger(name)
    if name and "Opponent_bot" in str(name):
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    if name and str(name).startswith("agent"):
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    return logger

logging.getLogger = _patched_get_logger

# Monkey-patch poke-env Mirror Move assertion bug
from poke_env.battle import pokemon as _pokemon_module

_original_available_moves_from_request = _pokemon_module.Pokemon.available_moves_from_request

def _patched_available_moves_from_request(self, request):
    try:
        return _original_available_moves_from_request(self, request)
    except (AssertionError, TypeError):
        return []

_pokemon_module.Pokemon.available_moves_from_request = _patched_available_moves_from_request

# Monkey-patch poke-env order_to_action infinite recursion bug
from poke_env.environment import singles_env as _singles_env_module

_original_order_to_action = _singles_env_module.SinglesEnv.order_to_action

def _patched_order_to_action(self, order, battle, **kwargs):
    kwargs.pop("strict", None)
    fake = kwargs.pop("fake", False)
    try:
        return _original_order_to_action(self, order, battle, fake)
    except (ValueError, RecursionError, AssertionError):
        return 10

_singles_env_module.SinglesEnv.order_to_action = _patched_order_to_action

# SB3
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

# SB3 MaskablePPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.battle import AbstractBattle, Battle
from poke_env.battle.status import Status as PokemonStatus
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import DefaultBattleOrder, RandomPlayer, SimpleHeuristicsPlayer, Player
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration

from tqdm import tqdm

logging.getLogger("poke_env.player").setLevel(logging.WARNING)


# -----------------------------
# Server Configuration
# -----------------------------
# LOCAL:    Use LocalhostServerConfiguration (requires local showdown server running)
# SHOWDOWN: Use ShowdownServerConfiguration  (requires registered account)
SERVER_CONFIG = LocalhostServerConfiguration


# -----------------------------
# Status helpers
# -----------------------------
def _is_frozen(mon) -> bool:
    if mon is None:
        return False
    try:
        status = mon.status
        if status is None:
            return False
        if status == PokemonStatus.FRZ:
            return True
        if hasattr(status, "name") and status.name == "FRZ":
            return True
        if hasattr(status, "value") and str(status.value).lower() == "frz":
            return True
    except Exception:
        pass
    return False


def _is_paralyzed(mon) -> bool:
    if mon is None:
        return False
    try:
        status = mon.status
        if status is None:
            return False
        if status == PokemonStatus.PAR:
            return True
        if hasattr(status, "name") and status.name == "PAR":
            return True
        if hasattr(status, "value") and str(status.value).lower() == "par":
            return True
    except Exception:
        pass
    return False


# -----------------------------
# Max Damage Player
# -----------------------------
class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


# -----------------------------
# Directories and constants
# -----------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

OBS_SIZE = 82
ACTION_SPACE_SIZE = 11
NET_ARCH = [256, 128, 64]
LEARNING_RATE = 1e-4
MAX_SPEED = 140

# --- Parallel training config ---
N_ENVS = 4
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10
ENT_COEF = 0.02
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95

STATUS_MAP = {
    None: 0.0,
    "brn": 0.2,
    "frz": 0.4,
    "par": 0.6,
    "psn": 0.8,
    "tox": 0.8,
    "slp": 1.0,
}

def model_path(name) -> Path:
    return MODEL_DIR / f"{name}.zip"


# -----------------------------
# Progress Callback
# -----------------------------
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, n_envs=1, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.episode_count = 0
        self.wins = 0
        self.losses = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="steps",
            dynamic_ncols=True
        )
        print(
            f"[Training] Total timesteps this run: {self.total_timesteps:,} | "
            f"Lifetime timesteps: {self.model._total_timesteps:,} | "
            f"n_envs={self.n_envs} | "
            f"effective_batch={N_STEPS * self.n_envs:,}"
        )

    def _on_step(self) -> bool:
        self.pbar.update(self.n_envs)

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1
                ep_info = info.get("episode", {})
                reward = ep_info.get("r", 0)
                if reward > 0.5:
                    self.wins += 1
                elif reward < -0.5:
                    self.losses += 1

        if self.episode_count > 0:
            wr = self.wins / self.episode_count
            self.pbar.set_postfix(
                battles=self.episode_count,
                W=self.wins,
                L=self.losses,
                WR=f"{wr:.1%}",
                refresh=False
            )
        return True

    def _on_training_end(self):
        self.pbar.close()


# -----------------------------
# Entropy Monitor Callback
# -----------------------------
class EntropyMonitorCallback(BaseCallback):
    ENTROPY_THRESHOLD = 0.5
    ENT_COEF_BOOST = 0.05
    ENT_COEF_NORMAL = ENT_COEF

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_count = 0
        self._boosted = False

    def _on_rollout_end(self):
        self.rollout_count += 1
        entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy is None:
            return

        actual_entropy = -entropy

        if actual_entropy < self.ENTROPY_THRESHOLD and not self._boosted:
            self.model.ent_coef = self.ENT_COEF_BOOST
            self._boosted = True
            print(f"\n[EntropyMonitor] Entropy collapsed ({actual_entropy:.3f}). Boosting ent_coef → {self.ENT_COEF_BOOST}")
        elif actual_entropy >= self.ENTROPY_THRESHOLD and self._boosted:
            self.model.ent_coef = self.ENT_COEF_NORMAL
            self._boosted = False
            print(f"\n[EntropyMonitor] Entropy recovered ({actual_entropy:.3f}). Restoring ent_coef → {self.ENT_COEF_NORMAL}")

    def _on_step(self) -> bool:
        return True


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

        self._all_types = list(self.gen_data.type_chart.keys())
        self._type_to_id = {t: i for i, t in enumerate(self._all_types)}
        self._type_count = max(len(self._all_types) - 1, 1)

        self._prev_opp_fainted: int = 0
        self._prev_my_fainted: int = 0
        self._prev_opp_hp: float = 1.0
        self._prev_my_hp: float = 1.0
        self._last_active_species: Optional[str] = None
        self._consec_frozen_wasted: int = 0
        self._consec_wasted_moves: int = 0
        self._frozen_opp_species: Optional[str] = None
        self._frozen_opp_last_hp: float = 1.0
        self._frozen_mons: set = set()

    def action_to_order(self, action, battle, fake=False, strict=True):
        if action == 10:
            action = -2
        return super().action_to_order(action, battle, fake=fake, strict=strict)

    def _get_speed(self, mon) -> float:
        if mon is None or mon.species is None:
            return 0.0
        try:
            speed = self.gen_data.pokedex[mon.species.lower()]["baseStats"]["spe"]
            return (speed / MAX_SPEED) * 2 - 1
        except (KeyError, TypeError):
            return 0.0

    def _get_type_advantage(self, attacker, defender) -> float:
        if attacker is None or defender is None:
            return 1.0
        best = 1.0
        for move in attacker.moves.values():
            try:
                mult = move.type.damage_multiplier(
                    defender.type_1,
                    defender.type_2,
                    type_chart=self.gen_data.type_chart
                )
                if mult > best:
                    best = mult
            except Exception:
                pass
        return best

    def _get_opp_move_effectiveness(self, battle) -> np.ndarray:
        features = np.zeros(4, dtype=np.float32)
        if battle.opponent_active_pokemon is None or battle.active_pokemon is None:
            return features

        revealed_moves = list(battle.opponent_active_pokemon.moves.values())
        for i, move in enumerate(revealed_moves[:4]):
            try:
                eff = move.type.damage_multiplier(
                    battle.active_pokemon.type_1,
                    battle.active_pokemon.type_2,
                    type_chart=self.gen_data.type_chart
                )
                features[i] = np.clip(eff - 1.0, -1.0, 1.0)
            except (AssertionError, TypeError):
                features[i] = 0.0

        return features

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any], env_id: int = 0) -> SingleAgentWrapper:
        agent_config = AccountConfiguration(f"agent_{env_id}", None)
        opponent_config = AccountConfiguration(f"Opponent_bot_{env_id}", None)

        env = cls(
            battle_format=config["battle_format"],
            log_level=30,
            open_timeout=None,
            strict=False,
            account_configuration1=agent_config,
            account_configuration2=opponent_config,
            server_configuration=LocalhostServerConfiguration,
        )
        opponent = RandomPlayer(start_listening=False, account_configuration=opponent_config)
        # opponent = MaxDamagePlayer(start_listening=False, account_configuration=opponent_config)
        # opponent = SimpleHeuristicsPlayer(start_listening=False, account_configuration=opponent_config)

        base_env = SingleAgentWrapper(env, opponent)
        return ActionMasker(base_env, mask_env)

    def _reset_battle_tracking(self):
        self._prev_opp_fainted = 0
        self._prev_my_fainted = 0
        self._prev_opp_hp = 1.0
        self._prev_my_hp = 1.0
        self._last_active_species = None
        self._consec_frozen_wasted = 0
        self._consec_wasted_moves = 0
        self._frozen_opp_species = None
        self._frozen_opp_last_hp = 1.0
        self._frozen_mons = set()

    def _get_team_hp_fraction(self, team) -> float:
        total_current = 0.0
        total_max = 0.0
        for mon in team.values():
            if not mon.fainted and mon.max_hp > 0:
                total_current += mon.current_hp
                total_max += mon.max_hp
        if total_max == 0:
            return 0.0
        return total_current / total_max

    def calc_reward(self, battle) -> float:
        reward = 0.0

        if battle.finished:
            if battle.won is None:
                reward = -0.3
            elif battle.won:
                reward = 1.0
            else:
                reward = -1.0
            self._reset_battle_tracking()
            return reward

        # --- Faint tracking ---
        opp_fainted_now = sum(p.fainted for p in battle.opponent_team.values())
        my_fainted_now = sum(p.fainted for p in battle.team.values())

        new_opp_faints = opp_fainted_now - self._prev_opp_fainted
        new_my_faints = my_fainted_now - self._prev_my_fainted

        reward += 0.02 * new_opp_faints
        reward -= 0.02 * new_my_faints

        self._prev_opp_fainted = opp_fainted_now
        self._prev_my_fainted = my_fainted_now

        # --- HP tracking — capture old values BEFORE updating ---
        old_opp_hp = self._prev_opp_hp
        old_my_hp = self._prev_my_hp

        opp_hp_now = self._get_team_hp_fraction(battle.opponent_team)
        my_hp_now = self._get_team_hp_fraction(battle.team)

        opp_hp_lost = old_opp_hp - opp_hp_now
        my_hp_lost = old_my_hp - my_hp_now

        reward += 0.002 * max(opp_hp_lost, 0) * 100
        reward -= 0.002 * max(my_hp_lost, 0) * 100

        self._prev_opp_hp = opp_hp_now
        self._prev_my_hp = my_hp_now

        # Per-turn stall penalty
        reward -= 0.001

        # --- Frozen opponent tracking ---
        for mon in battle.opponent_team.values():
            if _is_frozen(mon):
                self._frozen_mons.add(mon.species)
            elif mon.species in self._frozen_mons and not _is_frozen(mon):
                self._frozen_mons.discard(mon.species)

        opp_active = battle.opponent_active_pokemon

        if self._frozen_mons:
            if opp_active and opp_active.species in self._frozen_mons:
                current_hp = (opp_active.current_hp / opp_active.max_hp) if opp_active.max_hp > 0 else 0.0

                if opp_active.species != self._frozen_opp_species:
                    self._frozen_opp_species = opp_active.species
                    if self._frozen_opp_last_hp == 1.0:
                        self._frozen_opp_last_hp = current_hp
                else:
                    hp_dealt = self._frozen_opp_last_hp - current_hp
                    if hp_dealt > 0:
                        reward += 0.05 * hp_dealt * 100
                        self._consec_frozen_wasted = 0
                    else:
                        self._consec_frozen_wasted += 1
                        if self._consec_frozen_wasted > 3:
                            penalty = 0.05 * (self._consec_frozen_wasted - 2)
                            reward -= min(penalty, 1.0)
                    self._frozen_opp_last_hp = current_hp
            else:
                # Frozen mon is benched — penalize, do NOT reset tracking
                self._consec_frozen_wasted += 1
                penalty = 0.05 * self._consec_frozen_wasted
                reward -= min(penalty, 1.0)
        else:
            self._frozen_opp_species = None
            self._frozen_opp_last_hp = 1.0
            self._consec_frozen_wasted = 0

        # --- Wasted move penalty ---
        opp_hp_actually_dropped = (old_opp_hp - opp_hp_now) > 0.001
        my_hp_actually_dropped = (old_my_hp - my_hp_now) > 0.001

        if opp_hp_actually_dropped:
            self._consec_wasted_moves = 0
        elif not my_hp_actually_dropped and not battle.force_switch:
            self._consec_wasted_moves += 1
            if self._consec_wasted_moves > 3:
                penalty = 0.005 * (self._consec_wasted_moves - 3)
                reward -= min(penalty, 0.3)
        else:
            self._consec_wasted_moves = 0

        return reward

    def _encode_active_mon(self, mon) -> np.ndarray:
        features = np.zeros(5, dtype=np.float32)
        if mon is None:
            return features

        if mon.fainted or mon.max_hp == 0:
            features[0] = -1.0
        else:
            features[0] = (mon.current_hp / mon.max_hp) * 2 - 1

        status_key = mon.status.value if mon.status else None
        features[1] = STATUS_MAP.get(status_key, 0.0)

        if mon.species is not None:
            idx = self.species_to_id.get(mon.species.lower(), 0)
            features[2] = (idx / (len(self.species_to_id) - 1)) * 2 - 1

        if mon.type_1 is not None:
            features[3] = (self._type_to_id.get(mon.type_1.name, 0) / self._type_count) * 2 - 1

        if mon.type_2 is not None:
            features[4] = (self._type_to_id.get(mon.type_2.name, 0) / self._type_count) * 2 - 1

        return features

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
            opponent_identifier = np.zeros(pokemon_team, dtype=np.float32)
            self_status = np.zeros(pokemon_team, dtype=np.float32)
            opponent_status = np.zeros(pokemon_team, dtype=np.float32)
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

            for i, (_, mon) in enumerate(sorted(battle.team.items())):
                if mon is not None and mon.species is not None:
                    idx = self.species_to_id.get(mon.species.lower(), 0)
                    team_identifier[i] = (idx / (len(self.species_to_id) - 1)) * 2 - 1

            for i, (_, mon) in enumerate(sorted(battle.opponent_team.items())):
                if mon is not None and mon.species is not None:
                    idx = self.species_to_id.get(mon.species.lower(), 0)
                    opponent_identifier[i] = (idx / (len(self.species_to_id) - 1)) * 2 - 1

            for i, mon in enumerate(battle.team.values()):
                status_key = mon.status.value if mon.status else None
                self_status[i] = STATUS_MAP.get(status_key, 0.0)

            for i, mon in enumerate(battle.opponent_team.values()):
                status_key = mon.status.value if mon.status else None
                opponent_status[i] = STATUS_MAP.get(status_key, 0.0)

            if len(battle.available_moves) == 0 and len(battle.available_switches) == 0:
                special_case[0] = 1
            if battle.active_pokemon.must_recharge:
                special_case[1] = 1

            active_features = self._encode_active_mon(battle.active_pokemon)
            opp_active_features = self._encode_active_mon(battle.opponent_active_pokemon)

            own_speed = np.float32([self._get_speed(battle.active_pokemon)])
            opp_speed = np.float32([self._get_speed(battle.opponent_active_pokemon)])

            opp_move_eff = self._get_opp_move_effectiveness(battle)

            opp_active_frozen = np.float32([1.0 if _is_frozen(battle.opponent_active_pokemon) else 0.0])
            opp_active_paralyzed = np.float32([1.0 if _is_paralyzed(battle.opponent_active_pokemon) else 0.0])

            own_atk_boost = np.float32([
                battle.active_pokemon.boosts.get("atk", 0) / 6.0
                if battle.active_pokemon else 0.0
            ])
            own_spe_boost = np.float32([
                battle.active_pokemon.boosts.get("spa", 0) / 6.0
                if battle.active_pokemon else 0.0
            ])

            return np.float32(np.concatenate([
                moves_base_power,        # 4
                moves_dmg_multiplier,    # 4
                moves_pp_ratio,          # 4
                self_team_status,        # 6
                opponent_team_status,    # 6
                team_hp_ratio,           # 6
                opponent_hp_ratio,       # 6
                team_identifier,         # 6
                opponent_identifier,     # 6
                self_status,             # 6
                opponent_status,         # 6
                special_case,            # 2
                active_features,         # 5
                opp_active_features,     # 5
                own_speed,               # 1
                opp_speed,               # 1
                opp_move_eff,            # 4
                opp_active_frozen,       # 1
                opp_active_paralyzed,    # 1
                own_atk_boost,           # 1
                own_spe_boost,           # 1
            ]))                          # = 82 total

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
# Wrapper chain walker
# -----------------------------
def _get_custom_env(env) -> Optional[CustomEnv]:
    """Walk the wrapper chain until we find the CustomEnv instance."""
    obj = env
    while obj is not None:
        if isinstance(obj, CustomEnv):
            return obj
        obj = getattr(obj, 'env', None)
    return None


# -----------------------------
# Masking function
# -----------------------------
FROZEN_GRACE_PERIOD = 5

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

    opp = battle.opponent_active_pokemon
    opp_is_frozen = opp is not None and _is_frozen(opp)
    grace_expired = False

    if opp_is_frozen:
        custom_env = _get_custom_env(env)
        if custom_env is not None:
            grace_expired = custom_env._consec_frozen_wasted >= FROZEN_GRACE_PERIOD
        else:
            print("[mask_env] WARNING: Could not find CustomEnv in wrapper chain!")

    if not battle.force_switch or not battle.active_pokemon.fainted:
        for slot, move in enumerate(moves):
            if move in available_moves:
                if opp_is_frozen and grace_expired and move.base_power == 0:
                    continue
                action_mask[slot + move_offset] = 1

    if not (opp_is_frozen and grace_expired) or battle.force_switch:
        for slot, mon in enumerate(team):
            if mon in available_switches:
                action_mask[slot] = 1

    if not any(action_mask):
        action_mask[choose_default] = 1

    return action_mask


# -----------------------------
# Parallel env factory
# -----------------------------
def make_env_fn(env_id: int, seed: int = 0):
    def _init():
        set_random_seed(seed + env_id)
        env = CustomEnv.create_single_agent_env(
            {"battle_format": "gen1randombattle"},
            env_id=env_id
        )
        env = Monitor(env)
        return env
    return _init


def make_vec_env(n_envs: int = N_ENVS, use_subproc: bool = True):
    env_fns = [make_env_fn(env_id=i, seed=42) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        print(f"[VecEnv] Launching {n_envs} parallel environments (SubprocVecEnv)")
        return SubprocVecEnv(env_fns, start_method="spawn")
    else:
        print(f"[VecEnv] Using DummyVecEnv with {n_envs} env(s)")
        return DummyVecEnv(env_fns)


# -----------------------------
# Build model
# -----------------------------
def build_model(env, tensorboard_log="./tensorboard_logs/", model_name="model"):
    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=NET_ARCH)
    )


# -----------------------------
# Training functions
# -----------------------------
def train_new(model_name: str, timesteps: int, n_envs: int = N_ENVS, use_subproc: bool = True):
    path = model_path(model_name)
    if path.exists():
        print(f"Model '{model_name}' already exists! Delete it or choose another name.")
        return

    print(f"[Training Started] model={model_name}, timesteps={timesteps:,}, n_envs={n_envs}")
    train_env = make_vec_env(n_envs=n_envs, use_subproc=use_subproc)
    model = build_model(train_env, model_name=model_name)

    callbacks = CallbackList([
        ProgressCallback(timesteps, n_envs=n_envs),
        EntropyMonitorCallback(),
    ])

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=model_name
    )
    model.save(path)
    train_env.close()
    print(f"[Model Saved] {model_name}")


def train_continue(model_name: str, timesteps: int, n_envs: int = N_ENVS, use_subproc: bool = True):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found! Train a new model first.")
        return

    print(f"[Continuing Training] model={model_name}, additional timesteps={timesteps:,}, n_envs={n_envs}")
    train_env = make_vec_env(n_envs=n_envs, use_subproc=use_subproc)
    model = MaskablePPO.load(
        path,
        env=train_env,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(net_arch=NET_ARCH)
    )

    model.learning_rate = LEARNING_RATE
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    callbacks = CallbackList([
        ProgressCallback(timesteps, n_envs=n_envs),
        EntropyMonitorCallback(),
    ])

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=model_name
    )
    model.save(path)
    train_env.close()
    print(f"[Model Saved] {model_name}")


def eval_model(model_name: str, n_battles: int = 100):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found!")
        return

    print(f"[Evaluating] model={model_name}, battles={n_battles}")
    eval_env = make_vec_env(n_envs=1, use_subproc=False)
    model = MaskablePPO.load(
        path,
        env=eval_env,
        policy_kwargs=dict(net_arch=NET_ARCH)
    )

    wins, losses, draws = 0, 0, 0
    pbar = tqdm(total=n_battles, desc="Evaluating", unit="battles")

    obs = eval_env.reset()
    battles_done = 0

    while battles_done < n_battles:
        action_masks = get_action_masks(eval_env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        try:
            obs, reward, dones, infos = eval_env.step(action)
        except AssertionError:
            obs = eval_env.reset()
            continue

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                ep_reward = info.get("episode", {}).get("r", 0)
                if ep_reward > 0.5:
                    wins += 1
                elif ep_reward < -0.5:
                    losses += 1
                else:
                    draws += 1
                battles_done += 1
                pbar.update(1)
                pbar.set_postfix(W=wins, L=losses, D=draws, WR=f"{wins/battles_done:.1%}")

    pbar.close()
    eval_env.close()

    print(f"\n[Results] Battles: {n_battles} | Wins: {wins} | Losses: {losses} | Draws: {draws} | Win Rate: {wins/n_battles:.1%}")
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / n_battles}


# -----------------------------
# Play vs Human on local Showdown
# -----------------------------
#   7. Set HUMAN_USERNAME below to your browser username
#   8. Set MODE = "human" and run this script
#   9. The bot will log in and wait — challenge it from your browser!
# -----------------------------
async def play_vs_human(model_name: str, human_username: str, n_battles: int = 1):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found!")
        return

    print(f"[Human Mode] Loading {model_name}...")

    class AgentPlayer(Player):
        def __init__(self, model, **kwargs):
            super().__init__(**kwargs)
            self._model = model
            self._gen_data = GenData.from_gen(1)
            self._species_to_id = {
                name: i for i, name in enumerate(self._gen_data.pokedex.keys())
            }
            self._all_types = list(self._gen_data.type_chart.keys())
            self._type_to_id = {t: i for i, t in enumerate(self._all_types)}
            self._type_count = max(len(self._all_types) - 1, 1)

        def _get_speed(self, mon) -> float:
            if mon is None or mon.species is None:
                return 0.0
            try:
                speed = self._gen_data.pokedex[mon.species.lower()]["baseStats"]["spe"]
                return (speed / MAX_SPEED) * 2 - 1
            except (KeyError, TypeError):
                return 0.0

        def _encode_active_mon(self, mon) -> np.ndarray:
            features = np.zeros(5, dtype=np.float32)
            if mon is None:
                return features
            if mon.fainted or mon.max_hp == 0:
                features[0] = -1.0
            else:
                features[0] = (mon.current_hp / mon.max_hp) * 2 - 1
            status_key = mon.status.value if mon.status else None
            features[1] = STATUS_MAP.get(status_key, 0.0)
            if mon.species is not None:
                idx = self._species_to_id.get(mon.species.lower(), 0)
                features[2] = (idx / (len(self._species_to_id) - 1)) * 2 - 1
            if mon.type_1 is not None:
                features[3] = (self._type_to_id.get(mon.type_1.name, 0) / self._type_count) * 2 - 1
            if mon.type_2 is not None:
                features[4] = (self._type_to_id.get(mon.type_2.name, 0) / self._type_count) * 2 - 1
            return features

        def _embed(self, battle) -> np.ndarray:
            try:
                moves_n = 4
                pokemon_team = 6
                moves_base_power = -np.ones(moves_n)
                moves_dmg_multiplier = np.ones(moves_n)
                moves_pp_ratio = np.zeros(moves_n)
                team_hp_ratio = np.ones(pokemon_team)
                opponent_hp_ratio = np.ones(pokemon_team)
                opponent_team_status = np.ones(pokemon_team, dtype=np.float32)
                self_team_status = np.ones(pokemon_team, dtype=np.float32)
                team_identifier = np.zeros(pokemon_team, dtype=np.float32)
                opponent_identifier = np.zeros(pokemon_team, dtype=np.float32)
                self_status = np.zeros(pokemon_team, dtype=np.float32)
                opponent_status = np.zeros(pokemon_team, dtype=np.float32)
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
                                    type_chart=self._gen_data.type_chart
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

                for i, (_, mon) in enumerate(sorted(battle.team.items())):
                    if mon is not None and mon.species is not None:
                        idx = self._species_to_id.get(mon.species.lower(), 0)
                        team_identifier[i] = (idx / (len(self._species_to_id) - 1)) * 2 - 1

                for i, (_, mon) in enumerate(sorted(battle.opponent_team.items())):
                    if mon is not None and mon.species is not None:
                        idx = self._species_to_id.get(mon.species.lower(), 0)
                        opponent_identifier[i] = (idx / (len(self._species_to_id) - 1)) * 2 - 1

                for i, mon in enumerate(battle.team.values()):
                    status_key = mon.status.value if mon.status else None
                    self_status[i] = STATUS_MAP.get(status_key, 0.0)

                for i, mon in enumerate(battle.opponent_team.values()):
                    status_key = mon.status.value if mon.status else None
                    opponent_status[i] = STATUS_MAP.get(status_key, 0.0)

                if len(battle.available_moves) == 0 and len(battle.available_switches) == 0:
                    special_case[0] = 1
                if battle.active_pokemon and battle.active_pokemon.must_recharge:
                    special_case[1] = 1

                active_features = self._encode_active_mon(battle.active_pokemon)
                opp_active_features = self._encode_active_mon(battle.opponent_active_pokemon)
                own_speed = np.float32([self._get_speed(battle.active_pokemon)])
                opp_speed = np.float32([self._get_speed(battle.opponent_active_pokemon)])

                # Opponent move effectiveness
                opp_move_eff = np.zeros(4, dtype=np.float32)
                if battle.opponent_active_pokemon and battle.active_pokemon:
                    for i, move in enumerate(list(battle.opponent_active_pokemon.moves.values())[:4]):
                        try:
                            eff = move.type.damage_multiplier(
                                battle.active_pokemon.type_1,
                                battle.active_pokemon.type_2,
                                type_chart=self._gen_data.type_chart
                            )
                            opp_move_eff[i] = np.clip(eff - 1.0, -1.0, 1.0)
                        except Exception:
                            pass

                opp_active_frozen = np.float32([1.0 if _is_frozen(battle.opponent_active_pokemon) else 0.0])
                opp_active_paralyzed = np.float32([1.0 if _is_paralyzed(battle.opponent_active_pokemon) else 0.0])
                own_atk_boost = np.float32([battle.active_pokemon.boosts.get("atk", 0) / 6.0 if battle.active_pokemon else 0.0])
                own_spe_boost = np.float32([battle.active_pokemon.boosts.get("spa", 0) / 6.0 if battle.active_pokemon else 0.0])

                return np.float32(np.concatenate([
                    moves_base_power, moves_dmg_multiplier, moves_pp_ratio,
                    self_team_status, opponent_team_status,
                    team_hp_ratio, opponent_hp_ratio,
                    team_identifier, opponent_identifier,
                    self_status, opponent_status,
                    special_case, active_features, opp_active_features,
                    own_speed, opp_speed, opp_move_eff,
                    opp_active_frozen, opp_active_paralyzed,
                    own_atk_boost, own_spe_boost,
                ]))
            except Exception:
                return np.zeros(OBS_SIZE, dtype=np.float32)

        def _build_mask(self, battle) -> np.ndarray:
            action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
            if battle.active_pokemon is None:
                return action_mask

            available_moves = set(battle.available_moves)
            available_switches = set(battle.available_switches)
            moves = list(battle.active_pokemon.moves.values())
            team = list(battle.team.values())

            if len(available_moves) == 0 and len(available_switches) == 0:
                action_mask[10] = 1
                return action_mask
            if battle.active_pokemon.must_recharge:
                action_mask[10] = 1
                return action_mask

            opp_is_frozen = _is_frozen(battle.opponent_active_pokemon)

            if not battle.force_switch or not battle.active_pokemon.fainted:
                for slot, move in enumerate(moves):
                    if move in available_moves:
                        action_mask[slot + 6] = 1

            if not opp_is_frozen or battle.force_switch:
                for slot, mon in enumerate(team):
                    if mon in available_switches:
                        action_mask[slot] = 1

            if not any(action_mask):
                action_mask[10] = 1
            return action_mask

        async def choose_move(self, battle):
            print(f"[Bot] Turn {battle.turn}, force_switch={battle.force_switch}, must_recharge={battle.active_pokemon.must_recharge if battle.active_pokemon else 'N/A'}")
            print(f"[Bot] available_moves={[m.id for m in battle.available_moves]}")
            print(f"[Bot] available_switches={[m.species for m in battle.available_switches]}")

            obs = self._embed(battle)
            mask = self._build_mask(battle)

            print(f"[Bot] mask={mask}")

            obs_tensor = obs[np.newaxis, :]
            mask_tensor = mask[np.newaxis, :]
            action, _ = self._model.predict(obs_tensor, action_masks=mask_tensor, deterministic=True)
            action = int(action[0])

            print(f"[Bot] action={action}")

            available_moves = set(battle.available_moves)
            available_switches = set(battle.available_switches)
            moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
            team = list(battle.team.values())

            order = None

            if action == 10:
                order = self.choose_random_move(battle)
            elif action < 6:
                if action < len(team) and team[action] in available_switches:
                    order = self.create_order(team[action])
                else:
                    order = self.choose_random_move(battle)
            else:
                move_slot = action - 6
                if move_slot < len(moves) and moves[move_slot] in available_moves:
                    order = self.create_order(moves[move_slot])
                else:
                    order = self.choose_random_move(battle)

            print(f"[Bot] sending order={order}")
            return order

    # Load model for inference only — no env needed
    model = MaskablePPO.load(path, policy_kwargs=dict(net_arch=NET_ARCH))

    agent = AgentPlayer(
        model=model,
        battle_format="gen1randombattle",
        account_configuration=AccountConfiguration("PokemonBot", None),
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
    )

    print(f"[Human Mode] Bot logged in as 'PokemonBot' on localhost:8000")
    print(f"[Human Mode] Open http://localhost:8000 in your browser")
    print(f"[Human Mode] Log in as '{human_username}' and challenge 'PokemonBot'")
    print(f"[Human Mode] Waiting for {n_battles} battle(s)...")

    await agent.accept_challenges(human_username, n_battles)

    wins  = sum(1 for b in agent.battles.values() if b.won is True)
    losses = sum(1 for b in agent.battles.values() if b.won is False)
    draws  = sum(1 for b in agent.battles.values() if b.won is None)
    print(f"\n[Results] Bot: {wins}W / {losses}L / {draws}D")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Modes: "new" | "continue" | "eval" | "human"
    MODE = "eval"
    MODEL_NAME = "ParallelTest8"
    training_steps = 500000

    N_ENVS_RUN = 6
    USE_SUBPROC = True

    # Human mode settings
    # Your username as shown in the browser on localhost:8000
    HUMAN_USERNAME = "Grimgear76"
    N_HUMAN_BATTLES = 3

    if MODE == "new":
        train_new(MODEL_NAME, training_steps, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC)
    elif MODE == "continue":
        train_continue(MODEL_NAME, training_steps, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC)
    elif MODE == "eval":
        eval_model(MODEL_NAME, n_battles=100)
    elif MODE == "human":
        asyncio.run(play_vs_human(MODEL_NAME, HUMAN_USERNAME, N_HUMAN_BATTLES))

# Tensorboard: tensorboard --logdir ./tensorboard_logs/
# Older Python: .venv311\Scripts\python.exe -m tensorboard.main --logdir ./tensorboard_logs/
#



# Local Showdown setup:
#   git clone https://github.com/Grimgear76/pokemon-showdown.git
#   cd pokemon-showdown && npm install
#   cp config/config-example.js config/config.js
#   node pokemon-showdown start --no-security
#
# If npm install fails for pg:
#   Stop-Process -Name "node" -ErrorAction SilentlyContinue
#   npm install pg --save-dev
#
# Activate venv: .\.venv\Scripts\Activate.ps1