from typing import Any, Dict, Optional
from pathlib import Path
import multiprocessing

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
# Triggered by moves like Sky Attack that aren't in the available moves list,
# causing order_to_action to call itself recursively until stack overflow.
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
from poke_env import AccountConfiguration

from tqdm import tqdm

logging.getLogger("poke_env.player").setLevel(logging.WARNING)


# -----------------------------
# Status helpers
# -----------------------------
def _is_frozen(mon) -> bool:
    """Return True if the pokemon is frozen, using a robust multi-method check."""
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
    """Return True if the pokemon is paralyzed, using a robust multi-method check."""
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
        self._consec_frozen_wasted: int = 0   # turns wasted vs frozen opponent (switch OR bad move)
        self._consec_wasted_moves: int = 0    # turns where nothing happened at all
        self._frozen_opp_species: Optional[str] = None  # active frozen mon being tracked
        self._frozen_opp_last_hp: float = 1.0           # HP of active frozen mon last turn
        self._frozen_mons: set = set()                  # species of ALL currently frozen opponent mons

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
        """Returns the best damage multiplier any of attacker's moves has vs defender."""
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
            account_configuration2=opponent_config
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

        opp_fainted_now = sum(p.fainted for p in battle.opponent_team.values())
        my_fainted_now = sum(p.fainted for p in battle.team.values())

        new_opp_faints = opp_fainted_now - self._prev_opp_fainted
        new_my_faints = my_fainted_now - self._prev_my_fainted

        reward += 0.02 * new_opp_faints
        reward -= 0.02 * new_my_faints

        self._prev_opp_fainted = opp_fainted_now
        self._prev_my_fainted = my_fainted_now

        opp_hp_now = self._get_team_hp_fraction(battle.opponent_team)
        my_hp_now = self._get_team_hp_fraction(battle.team)

        opp_hp_lost = self._prev_opp_hp - opp_hp_now
        my_hp_lost = self._prev_my_hp - my_hp_now

        reward += 0.002 * max(opp_hp_lost, 0) * 100
        reward -= 0.002 * max(my_hp_lost, 0) * 100

        self._prev_opp_hp = opp_hp_now
        self._prev_my_hp = my_hp_now

        # Per-turn stall penalty up to -1.0 after max 1000 turns
        reward -= 0.001

        # --- Frozen opponent tracking ---
        # Maintain registry of ALL frozen opponent mons across the whole battle.
        # This persists across switches so the penalty counter cannot be reset by
        # rotating party members — switching away from a frozen mon is just as bad
        # as attacking it ineffectively.
        for mon in battle.opponent_team.values():
            if _is_frozen(mon):
                self._frozen_mons.add(mon.species)
            elif mon.species in self._frozen_mons and not _is_frozen(mon):
                # Mon thawed naturally or fainted — remove from registry
                self._frozen_mons.discard(mon.species)

        opp_active = battle.opponent_active_pokemon

        if self._frozen_mons:
            if opp_active and opp_active.species in self._frozen_mons:
                # Frozen mon is active — check if we dealt damage this turn
                current_hp = (opp_active.current_hp / opp_active.max_hp) if opp_active.max_hp > 0 else 0.0

                if opp_active.species != self._frozen_opp_species:
                    # Switched back to frozen target — reset HP baseline but carry
                    # the wasted-turn counter forward so switching can't game the penalty
                    self._frozen_opp_species = opp_active.species
                    self._frozen_opp_last_hp = current_hp
                else:
                    hp_dealt = self._frozen_opp_last_hp - current_hp
                    if hp_dealt > 0:
                        # Dealt damage — reward and reset counter
                        reward += 0.05 * hp_dealt * 100
                        self._consec_frozen_wasted = 0
                    else:
                        # No damage dealt while frozen mon is active — escalate  3 turn grace period
                        self._consec_frozen_wasted += 1
                        if self._consec_frozen_wasted > 3:
                            penalty = 0.02 * (self._consec_frozen_wasted - 2)
                            reward -= min(penalty, 0.5)
                    self._frozen_opp_last_hp = current_hp
            else:
                # Frozen mon exists but agent switched it to the bench — penalize
                # every turn we're not finishing it, counter keeps accumulating
                self._consec_frozen_wasted += 1
                penalty = 0.02 * self._consec_frozen_wasted
                reward -= min(penalty, 0.5)
                # Clear active tracker so HP baseline resets correctly when we switch back
                self._frozen_opp_species = None
        else:
            # No frozen mons on opponent's team — clear all tracking
            self._frozen_opp_species = None
            self._consec_frozen_wasted = 0

        # --- Wasted move penalty ---
        # Escalating penalty for repeatedly doing nothing useful:
        # no HP change on either side and no force switch (e.g. failed status moves,
        # maxed stat boosts, spamming Thunder Wave on paralyzed target).
        # Grace period of 3 turns before any penalty kicks in.
        opp_hp_actually_dropped = (self._prev_opp_hp - opp_hp_now) > 0.001
        my_hp_actually_dropped = (self._prev_my_hp - my_hp_now) > 0.001

        if opp_hp_actually_dropped:
            self._consec_wasted_moves = 0
        elif not my_hp_actually_dropped and not battle.force_switch:
            self._consec_wasted_moves += 1
            if self._consec_wasted_moves > 3:  # 3-turn grace period
                penalty = 0.005 * (self._consec_wasted_moves - 3)
                penalty = min(penalty, 0.3)    # soft cap
                reward -= penalty
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

            opp_move_eff = self._get_opp_move_effectiveness(battle)  # 4

            opp_active_frozen = np.float32([1.0 if _is_frozen(battle.opponent_active_pokemon) else 0.0])       # 1
            opp_active_paralyzed = np.float32([1.0 if _is_paralyzed(battle.opponent_active_pokemon) else 0.0]) # 1

            # Stat boosts normalized -6..+6 → -1..1 so agent knows when buffs are maxed
            own_atk_boost = np.float32([
                battle.active_pokemon.boosts.get("atk", 0) / 6.0
                if battle.active_pokemon else 0.0
            ])  # 1
            own_spe_boost = np.float32([
                battle.active_pokemon.boosts.get("spa", 0) / 6.0
                if battle.active_pokemon else 0.0
            ])  # 1

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

    # Truly no options at all — must use default
    if len(available_moves) == 0 and len(available_switches) == 0:
        action_mask[choose_default] = 1
        return action_mask

    # Must recharge — must use default
    if battle.active_pokemon.must_recharge:
        action_mask[choose_default] = 1
        return action_mask

    # Enable move actions
    if not battle.force_switch or not battle.active_pokemon.fainted:
        for slot, move in enumerate(moves):
            if move in available_moves:
                action_mask[slot + move_offset] = 1

    # Enable all valid switches — reward shaping handles frozen opponent stalling
    for slot, mon in enumerate(team):
        if mon in available_switches:
            action_mask[slot] = 1

    # choose_default only as absolute last resort
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
# Run
# -----------------------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # required on Windows

    MODE = "new"             # "new" | "continue" | "eval"
    MODEL_NAME = "ParallelTest8"
    training_steps = 500000

    N_ENVS_RUN = 6       # 1-8, lower = less CPU usage
    USE_SUBPROC = True   # False if using only 1 env

    if MODE == "new":
        train_new(MODEL_NAME, training_steps, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC)
    elif MODE == "continue":
        train_continue(MODEL_NAME, training_steps, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC)
    elif MODE == "eval":
        eval_model(MODEL_NAME, n_battles=100)

# Tensorboard command: tensorboard --logdir ./tensorboard_logs/

# if running on newer python version that doesnt support tensorboard yet
# .venv311\Scripts\python.exe -m tensorboard.main --logdir ./tensorboard_logs/