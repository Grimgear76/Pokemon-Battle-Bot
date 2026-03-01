import logging
import numpy as np
from typing import Any, Dict, Optional

from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.battle import AbstractBattle, Battle
from poke_env.battle.status import Status as PokemonStatus
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from tqdm import tqdm

from constants import (
    OBS_SIZE, ACTION_SPACE_SIZE, NET_ARCH, MAX_SPEED,
    N_ENVS, N_STEPS, ENT_COEF, STATUS_MAP
)

# -----------------------------
# Logging patches
# -----------------------------
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

# -----------------------------
# Monkey-patch: Mirror Move assertion bug
# -----------------------------
from poke_env.battle import pokemon as _pokemon_module

_original_available_moves_from_request = _pokemon_module.Pokemon.available_moves_from_request

def _patched_available_moves_from_request(self, request):
    try:
        return _original_available_moves_from_request(self, request)
    except (AssertionError, TypeError):
        return []

_pokemon_module.Pokemon.available_moves_from_request = _patched_available_moves_from_request

# -----------------------------
# Monkey-patch: order_to_action infinite recursion bug
# -----------------------------
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

logging.getLogger("poke_env.player").setLevel(logging.WARNING)


# -----------------------------
# Status helpers
# -----------------------------
def _has_status(mon, status_enum) -> bool:
    if mon is None:
        return False
    try:
        status = mon.status
        if status is None:
            return False
        if status == status_enum:
            return True
        if hasattr(status, "name") and status.name == status_enum.name:
            return True
    except Exception:
        pass
    return False

def _is_frozen(mon) -> bool:
    return _has_status(mon, PokemonStatus.FRZ)

def _is_paralyzed(mon) -> bool:
    return _has_status(mon, PokemonStatus.PAR)

def _is_asleep(mon) -> bool:
    return _has_status(mon, PokemonStatus.SLP)

def _is_burned(mon) -> bool:
    return _has_status(mon, PokemonStatus.BRN)

def _is_poisoned(mon) -> bool:
    """Returns True for both regular poison (PSN) and toxic (TOX)."""
    if mon is None:
        return False
    try:
        status = mon.status
        if status is None:
            return False
        if status in (PokemonStatus.PSN, PokemonStatus.TOX):
            return True
        if hasattr(status, "name") and status.name in ("PSN", "TOX"):
            return True
    except Exception:
        pass
    return False

def _encode_status_flags(mon) -> np.ndarray:
    """
    Returns a 5-element float32 array of binary status flags:
    [asleep, frozen, paralyzed, burned, poisoned]
    """
    return np.float32([
        1.0 if _is_asleep(mon)    else 0.0,
        1.0 if _is_frozen(mon)    else 0.0,
        1.0 if _is_paralyzed(mon) else 0.0,
        1.0 if _is_burned(mon)    else 0.0,
        1.0 if _is_poisoned(mon)  else 0.0,
    ])


def _encode_boosts(mon) -> np.ndarray:
    """
    Returns a 5-element float32 array of stat boosts normalized to [-1, 1]:
    [atk, def, spe, spa, spd]
    Returns zeros if mon is None.
    """
    if mon is None:
        return np.zeros(5, dtype=np.float32)
    boosts = mon.boosts if mon.boosts else {}
    return np.float32([
        boosts.get("atk", 0) / 6.0,
        boosts.get("def", 0) / 6.0,
        boosts.get("spe", 0) / 6.0,
        boosts.get("spa", 0) / 6.0,
        boosts.get("spd", 0) / 6.0,
    ])


# Canonical Gen 2 type list — 18 types, fixed order for consistent one-hot encoding.
# Must never be reordered between training runs.
GEN2_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "???",
]
_TYPE_TO_IDX = {t.lower(): i for i, t in enumerate(GEN2_TYPES)}
N_TYPES = len(GEN2_TYPES)  # 18


def _type_onehot(type_obj) -> np.ndarray:
    """Return an 18-dim one-hot vector for a poke-env Type object, or all-zeros if None."""
    vec = np.zeros(N_TYPES, dtype=np.float32)
    if type_obj is None:
        return vec
    idx = _TYPE_TO_IDX.get(type_obj.name.lower(), -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def _encode_active_mon(mon) -> np.ndarray:
    """
    Returns a 38-element float32 array:
      [hp(1), status(1), type1_onehot(18), type2_onehot(18)]
    type2 is all-zeros for mono-type Pokémon.
    """
    hp = np.float32([-1.0 if (mon is None or mon.fainted or mon.max_hp == 0)
                     else (mon.current_hp / mon.max_hp) * 2 - 1])
    status_key = (mon.status.value if mon.status else None) if mon is not None else None
    status = np.float32([STATUS_MAP.get(status_key, 0.0)])
    type1 = _type_onehot(mon.type_1 if mon is not None else None)
    type2 = _type_onehot(mon.type_2 if mon is not None else None)
    return np.concatenate([hp, status, type1, type2])  # 1+1+18+18 = 38


# -----------------------------
# Max Damage Player
# -----------------------------
from poke_env.player import Player

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


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
        # Gen 2 data
        self.gen_data = GenData.from_gen(2)
        self.species_to_id = {
            name: i for i, name in enumerate(self.gen_data.pokedex.keys())
        }

        # Build item ID lookup from Gen 2 data
        try:
            all_items = list(self.gen_data.items.keys())
        except AttributeError:
            all_items = []
        self._item_to_id = {name: i + 1 for i, name in enumerate(all_items)}
        self._item_count = max(len(self._item_to_id), 1)

        self._prev_opp_fainted: int = 0
        self._prev_my_fainted: int = 0
        self._prev_opp_hp: float = 1.0
        self._prev_my_hp: float = 1.0
        self._consec_wasted_moves: int = 0
        self._consec_wasted_switches: int = 0
        self._last_action_was_switch: bool = False

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

    def _get_item_id(self, mon) -> float:
        """Returns normalized item ID in [-1, 1]. 0.0 means no item or unknown."""
        if mon is None:
            return 0.0
        try:
            item = mon.item
            if item is None or item == "" or item == "unknown_item":
                return 0.0
            item_id = self._item_to_id.get(item.lower(), 0)
            if item_id == 0:
                return 0.0
            return (item_id / self._item_count) * 2 - 1
        except Exception:
            return 0.0

    def _get_type_advantage(self, attacker, defender) -> float:
        if attacker is None or defender is None:
            return 1.0
        best = 1.0
        for move in attacker.moves.values():
            try:
                mult = move.type.damage_multiplier(
                    defender.type_1, defender.type_2,
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
                    battle.active_pokemon.type_1, battle.active_pokemon.type_2,
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

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        #opponent = RandomPlayer(start_listening=False, account_configuration=opponent_config)
        #opponent = MaxDamagePlayer(start_listening=False, account_configuration=opponent_config)
        opponent = SimpleHeuristicsPlayer(start_listening=False, account_configuration=opponent_config)
        #----------------------------------------------------------------------------------------------------------------------------------------------------------

        base_env = SingleAgentWrapper(env, opponent)
        return ActionMasker(base_env, mask_env)

    def _reset_battle_tracking(self):
        self._prev_opp_fainted = 0
        self._prev_my_fainted = 0
        self._prev_opp_hp = 1.0
        self._prev_my_hp = 1.0
        self._consec_wasted_moves = 0
        self._consec_wasted_switches = 0
        self._last_action_was_switch = False

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
                reward = -0.5
            elif battle.won:
                reward = 1.0
            else:
                reward = -1.0
            self._reset_battle_tracking()
            return reward

        # --- Faint tracking ---
        # KO reward: +0.04 per opponent fainted, -0.08 per own fainted
        # A full 6-0 sweep = +0.24 in KO rewards, meaningful vs win=+1.0
        opp_fainted_now = sum(p.fainted for p in battle.opponent_team.values())
        my_fainted_now  = sum(p.fainted for p in battle.team.values())

        new_opp_faints = opp_fainted_now - self._prev_opp_fainted
        new_my_faints  = my_fainted_now  - self._prev_my_fainted

        reward += 0.04 * new_opp_faints
        reward -= 0.04 * new_my_faints

        self._prev_opp_fainted = opp_fainted_now
        self._prev_my_fainted  = my_fainted_now

        # --- HP tracking ---
        # Damage reward: 50% HP dealt = +0.001  (0.002 * 0.5 = 0.001)
        old_opp_hp = self._prev_opp_hp
        old_my_hp  = self._prev_my_hp

        opp_hp_now = self._get_team_hp_fraction(battle.opponent_team)
        my_hp_now  = self._get_team_hp_fraction(battle.team)

        opp_hp_lost = old_opp_hp - opp_hp_now
        my_hp_lost  = old_my_hp  - my_hp_now

        if opp_hp_lost > 0.001:
            reward += 0.002 * opp_hp_lost * 100
        if my_hp_lost > 0.001:
            reward -= 0.002 * my_hp_lost * 100

        self._prev_opp_hp = opp_hp_now
        self._prev_my_hp  = my_hp_now

        opp_hp_actually_dropped = opp_hp_lost > 0.001
        my_hp_actually_dropped  = my_hp_lost  > 0.001

        # --- Per-turn stall penalty ---
        reward -= 0.001

        # --- Switch penalty ---
        # Grace of 3 consecutive non-damaging switches before penalty kicks in.
        if self._last_action_was_switch and not battle.force_switch and not opp_hp_actually_dropped:
            self._consec_wasted_switches += 1
            if self._consec_wasted_switches > 3:
                reward -= 0.01 * (self._consec_wasted_switches - 3)
        else:
            self._consec_wasted_switches = 0

        # --- Wasted move penalty ---
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
                if mon.fainted: self_team_status[i] = -1.0
            for i, mon in enumerate(battle.opponent_team.values()):
                if mon.fainted: opponent_team_status[i] = -1.0
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

            # One-hot typed active mon features (38 each)
            active_features     = _encode_active_mon(battle.active_pokemon)
            opp_active_features = _encode_active_mon(battle.opponent_active_pokemon)

            own_speed    = np.float32([self._get_speed(battle.active_pokemon)])
            opp_speed    = np.float32([self._get_speed(battle.opponent_active_pokemon)])
            opp_move_eff = self._get_opp_move_effectiveness(battle)

            # All boosts normalized to [-1, 1] via /6
            own_boosts = _encode_boosts(battle.active_pokemon)
            opp_boosts = _encode_boosts(battle.opponent_active_pokemon)

            # Gen 2: held item IDs
            own_item_id = np.float32([self._get_item_id(battle.active_pokemon)])
            opp_item_id = np.float32([self._get_item_id(battle.opponent_active_pokemon)])

            own_status_flags = _encode_status_flags(battle.active_pokemon)
            opp_status_flags = _encode_status_flags(battle.opponent_active_pokemon)

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
                active_features,         # 38  [hp, status, type1_onehot(18), type2_onehot(18)]
                opp_active_features,     # 38  [hp, status, type1_onehot(18), type2_onehot(18)]
                own_speed,               # 1
                opp_speed,               # 1
                opp_move_eff,            # 4
                own_boosts,              # 5   [atk, def, spe, spa, spd]
                opp_boosts,              # 5   [atk, def, spe, spa, spd]
                own_item_id,             # 1
                opp_item_id,             # 1
                own_status_flags,        # 5   [slp, frz, par, brn, psn]
                opp_status_flags,        # 5   [slp, frz, par, brn, psn]
            ]))
            # Total: 4+4+4+6+6+6+6+6+6+6+6+2+38+38+1+1+4+5+5+1+1+5+5 = 166

        except AssertionError:
            return np.zeros(OBS_SIZE, dtype=np.float32)

    def reset(self, *args, **kwargs):
        self._reset_battle_tracking()
        return super().reset(*args, **kwargs)

    def close(self):
        super().close()
        print("[Environment Closed]")

    def step(self, action):
        if isinstance(action, dict):
            raw_action = next(iter(action.values()))
        else:
            raw_action = action
        self._last_action_was_switch = (raw_action < 6)
        return super().step(action)


# -----------------------------
# Wrapper chain walker
# -----------------------------
def _get_custom_env(env) -> Optional[CustomEnv]:
    obj = env
    while obj is not None:
        if isinstance(obj, CustomEnv):
            return obj
        obj = getattr(obj, 'env', None)
    return None


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

    # --- Own status checks ---
    own_mon = battle.active_pokemon
    own_incapacitated = _is_asleep(own_mon) or _is_frozen(own_mon)

    # --- Build move mask ---
    if not battle.force_switch or not battle.active_pokemon.fainted:
        for slot, move in enumerate(moves):
            if move in available_moves:
                action_mask[slot + move_offset] = 1

    # --- Build switch mask ---
    # Always allow switching when incapacitated (frozen/asleep) or force switch
    allow_switch = battle.force_switch or own_incapacitated or bool(available_moves)
    if allow_switch:
        for slot, mon in enumerate(team):
            if mon in available_switches:
                action_mask[slot] = 1

    if not any(action_mask):
        action_mask[choose_default] = 1

    return action_mask


# -----------------------------
# Parallel env factory
# -----------------------------
def make_env_fn(env_id: int, seed: int = 0, battle_format: str = "gen2randombattle"):
    def _init():
        set_random_seed(seed + env_id)
        env = CustomEnv.create_single_agent_env(
            {"battle_format": battle_format},
            env_id=env_id
        )
        env = Monitor(env)
        return env
    return _init


def make_vec_env(n_envs: int = N_ENVS, use_subproc: bool = True, battle_format: str = "gen2randombattle"):
    env_fns = [make_env_fn(env_id=i, seed=42, battle_format=battle_format) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        print(f"[VecEnv] Launching {n_envs} parallel environments (SubprocVecEnv) | format={battle_format}")
        return SubprocVecEnv(env_fns, start_method="spawn")
    else:
        print(f"[VecEnv] Using DummyVecEnv with {n_envs} env(s) | format={battle_format}")
        return DummyVecEnv(env_fns)