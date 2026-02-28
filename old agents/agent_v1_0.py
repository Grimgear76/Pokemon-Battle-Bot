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

# Monkey-patch poke-env Mirror Move assertion bug
from poke_env.battle import pokemon as _pokemon_module

_original_available_moves_from_request = _pokemon_module.Pokemon.available_moves_from_request

def _patched_available_moves_from_request(self, request):
    try:
        return _original_available_moves_from_request(self, request)
    except (AssertionError, TypeError):
        return []

_pokemon_module.Pokemon.available_moves_from_request = _patched_available_moves_from_request

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
from poke_env.player import DefaultBattleOrder, RandomPlayer, SimpleHeuristicsPlayer, Player
from poke_env import AccountConfiguration

from tqdm import tqdm

logging.getLogger("poke_env.player").setLevel(logging.WARNING)
logging.getLogger("agent").setLevel(logging.CRITICAL)


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

OBS_SIZE = 78  # 74 + 4 opponent revealed move effectiveness features
ACTION_SPACE_SIZE = 11
NET_ARCH = [256, 128, 64]
LEARNING_RATE = 0.0001
MAX_SPEED = 140  # Electrode is fastest in gen 1

# Status condition encoding
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

        # Precompute type encoding
        self._all_types = list(self.gen_data.type_chart.keys())
        self._type_to_id = {t: i for i, t in enumerate(self._all_types)}
        self._type_count = max(len(self._all_types) - 1, 1)

        # Track previous battle state for step rewards
        self._prev_opp_fainted: int = 0
        self._prev_my_fainted: int = 0
        self._prev_opp_hp: float = 1.0
        self._prev_my_hp: float = 1.0
        self._last_active_species: Optional[str] = None

    def action_to_order(self, action, battle, fake=False, strict=True):
        if action == 10:
            action = -2
        return super().action_to_order(action, battle, fake=fake, strict=strict)

    def _get_speed(self, mon) -> float:
        """Return normalized speed for a pokemon using poke-env pokedex data."""
        if mon is None or mon.species is None:
            return 0.0
        try:
            speed = self.gen_data.pokedex[mon.species.lower()]["baseStats"]["spe"]
            return (speed / MAX_SPEED) * 2 - 1
        except (KeyError, TypeError):
            return 0.0

    def _get_opp_move_effectiveness(self, battle) -> np.ndarray:
        """
        Encode up to 4 revealed opponent moves as effectiveness against
        our active pokemon. Returns 0.0 for unrevealed move slots.
        Normalized to [-1, 1] by subtracting 1.0 and clipping.
        """
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

    def _opp_is_frozen(self, battle) -> bool:
        """Check if opponent active pokemon is frozen."""
        if battle.opponent_active_pokemon is None:
            return False
        try:
            status = battle.opponent_active_pokemon.status
            return status is not None and status.value == "frz"
        except Exception:
            return False

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

    def _get_team_hp_fraction(self, team) -> float:
        """Total current HP fraction across all pokemon on a side."""
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

        # --- End of battle ---
        if battle.finished:
            if battle.won is None:
                reward = 0.0
            elif battle.won:
                reward = 4.0
            else:
                reward = -4.0

            self._reset_battle_tracking()
            return reward

        # --- Per-step faint rewards ---
        opp_fainted_now = sum(p.fainted for p in battle.opponent_team.values())
        my_fainted_now = sum(p.fainted for p in battle.team.values())

        new_opp_faints = opp_fainted_now - self._prev_opp_fainted
        new_my_faints = my_fainted_now - self._prev_my_fainted

        reward += 0.08 * new_opp_faints
        reward -= 0.08 * new_my_faints

        self._prev_opp_fainted = opp_fainted_now
        self._prev_my_fainted = my_fainted_now

        # --- HP delta reward (dense signal every turn) ---
        opp_hp_now = self._get_team_hp_fraction(battle.opponent_team)
        my_hp_now = self._get_team_hp_fraction(battle.team)

        opp_hp_lost = self._prev_opp_hp - opp_hp_now
        my_hp_lost = self._prev_my_hp - my_hp_now

        reward += 0.002 * max(opp_hp_lost, 0) * 100
        reward -= 0.002 * max(my_hp_lost, 0) * 100

        self._prev_opp_hp = opp_hp_now
        self._prev_my_hp = my_hp_now

        # --- Freeze swap penalty ---
        # Penalize switching when opponent is frozen to prevent swap farming
        if battle.active_pokemon is not None:
            current_species = battle.active_pokemon.species
            if (self._last_active_species is not None and
                    current_species != self._last_active_species and
                    self._opp_is_frozen(battle)):
                reward -= 0.04

            self._last_active_species = current_species

        return reward

    def _encode_active_mon(self, mon) -> np.ndarray:
        """Encode active pokemon into 5 features: [hp, status, species_id, type1, type2]"""
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

            # Opponent revealed move effectiveness against our active pokemon
            opp_move_eff = self._get_opp_move_effectiveness(battle)  # 4

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
            ]))                          # = 78 total

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
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=LEARNING_RATE,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(net_arch=NET_ARCH)
    )
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
    model = MaskablePPO.load(
        path,
        env=train_env,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(net_arch=NET_ARCH)
    )

    # Force learning rate override on optimizer directly
    model.learning_rate = LEARNING_RATE
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

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
    model = MaskablePPO.load(
        path,
        env=eval_env,
        policy_kwargs=dict(net_arch=NET_ARCH)
    )

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
    MODE = "new"             # "new" | "continue" | "eval"
    MODEL_NAME = "RewardTest11"
    training_steps = 500000

    if MODE == "new":
        train_new(MODEL_NAME, training_steps)
    elif MODE == "continue":
        train_continue(MODEL_NAME, training_steps)
    elif MODE == "eval":
        eval_model(MODEL_NAME, n_battles=100)

# For tensorboard run command in separate terminal:      tensorboard --logdir ./tensorboard_logs/