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
    N_ENVS, N_STEPS, ENT_COEF, STATUS_MAP,
    SPECIES_NUM_SCALE,
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
    Returns a 37-element float32 array:
      [hp(1), type1_onehot(18), type2_onehot(18)]
    type2 is all-zeros for mono-type Pokemon.

    Status scalar removed — it was a duplicate of own_status_flags / opp_status_flags
    already present in the main observation, wasting 2 dimensions per call.
    """
    hp = np.float32([-1.0 if (mon is None or mon.fainted or mon.max_hp == 0)
                     else (mon.current_hp / mon.max_hp) * 2 - 1])
    type1 = _type_onehot(mon.type_1 if mon is not None else None)
    type2 = _type_onehot(mon.type_2 if mon is not None else None)
    return np.concatenate([hp, type1, type2])  # 1+18+18 = 37


# -----------------------------
# Max Damage Player
# -----------------------------
from poke_env.player import Player

class MaxDamagePlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_data = GenData.from_gen(2)

    def _is_move_useful(self, move, mon, battle) -> bool:
        boosts = mon.boosts if mon.boosts is not None else {}
        if move.id == "rest":
            if _is_asleep(mon): return False
            if mon.max_hp > 0 and (mon.current_hp / mon.max_hp) > 0.8: return False
            if mon.status is None and mon.max_hp > 0 and (mon.current_hp / mon.max_hp) > 0.5: return False
        if move.id == "sleeptalk" and not _is_asleep(mon): return False
        if getattr(move, 'heal', 0) and move.id != "rest":
            if mon.max_hp > 0 and (mon.current_hp / mon.max_hp) > 0.8: return False
        if getattr(move, 'status', None) and battle.opponent_active_pokemon is not None:
            if battle.opponent_active_pokemon.status is not None: return False
        if move.base_power == 0 or move.base_power is None:
            move_id = move.id
            if move_id in ("agility", "rocksmash") and boosts.get("spe", 0) >= 6: return False
            if move_id in ("swordsdance", "meditate", "sharpen") and boosts.get("atk", 0) >= 6: return False
            if move_id == "growth" and boosts.get("spa", 0) >= 6 and boosts.get("spd", 0) >= 6: return False
            if move_id in ("nastyplot", "chargebeam") and boosts.get("spa", 0) >= 6: return False
            if move_id in ("defensecurl", "harden", "withdraw") and boosts.get("def", 0) >= 6: return False
            if move_id == "curse" and boosts.get("atk", 0) >= 6 and boosts.get("def", 0) >= 6: return False
        return True

    def choose_move(self, battle):
        mon = battle.active_pokemon
        damaging_moves = [m for m in battle.available_moves if m.base_power and m.base_power > 0]
        if damaging_moves:
            def effective_power(move):
                if battle.opponent_active_pokemon is None: return move.base_power
                try:
                    mult = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=self.gen_data.type_chart)
                    return move.base_power * mult
                except Exception: return move.base_power
            return self.create_order(max(damaging_moves, key=effective_power))
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        if mon is not None:
            useful_moves = [m for m in battle.available_moves if self._is_move_useful(m, mon, battle)]
            if useful_moves: return self.create_order(useful_moves[0])
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
        self._custom_envs = []
        self._prev_w = []
        self._prev_l = []

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps", dynamic_ncols=True)
        print(
            f"[Training] Total timesteps this run: {self.total_timesteps:,} | "
            f"Lifetime timesteps: {self.model._total_timesteps:,} | "
            f"n_envs={self.n_envs} | "
            f"effective_batch={N_STEPS * self.n_envs:,}"
        )
        self._custom_envs = []
        self._use_counters = False
        try:
            for env in self.training_env.envs:
                inner = env
                while inner is not None:
                    if hasattr(inner, 'eval_wins'):
                        self._custom_envs.append(inner)
                        break
                    inner = getattr(inner, 'env', None)
                else:
                    self._custom_envs.append(None)
            self._use_counters = any(c is not None for c in self._custom_envs)
        except AttributeError:
            pass  # SubprocVecEnv — no .envs attribute
        self._prev_w = [c.eval_wins   if c else 0 for c in self._custom_envs]
        self._prev_l = [c.eval_losses if c else 0 for c in self._custom_envs]

    def _on_step(self) -> bool:
        self.pbar.update(self.n_envs)
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                if self._use_counters:
                    self.wins = sum(
                        (self._custom_envs[j].eval_wins   - self._prev_w[j])
                        for j in range(len(self._custom_envs)) if self._custom_envs[j]
                    )
                    self.losses = sum(
                        (self._custom_envs[j].eval_losses - self._prev_l[j])
                        for j in range(len(self._custom_envs)) if self._custom_envs[j]
                    )
                else:
                    info = infos[i] if i < len(infos) else {}
                    terminal_reward = info.get("terminal_reward", None)
                    if terminal_reward is not None:
                        if terminal_reward > 0.5:
                            self.wins += 1
                        elif terminal_reward < -0.5:
                            self.losses += 1
                    else:
                        ep_reward = info.get("episode", {}).get("r", 0)
                        if ep_reward > 2.0:
                            self.wins += 1
                        elif ep_reward < -2.0:
                            self.losses += 1
        if self.episode_count > 0:
            wr = self.wins / self.episode_count
            self.pbar.set_postfix(battles=self.episode_count, W=self.wins, L=self.losses, WR=f"{wr:.1%}", refresh=False)
        return True

    def _on_training_end(self):
        self.pbar.close()


# -----------------------------
# Entropy Monitor Callback
# -----------------------------
class EntropyMonitorCallback(BaseCallback):
    # Threshold raised from 0.5 to 1.0.
    ENTROPY_THRESHOLD = 1.0
    ENT_COEF_BOOST = 0.05
    ENT_COEF_NORMAL = ENT_COEF

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_count = 0
        self._boosted = False

    def _on_rollout_end(self):
        self.rollout_count += 1
        entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy is None: return
        actual_entropy = -entropy
        if actual_entropy < self.ENTROPY_THRESHOLD and not self._boosted:
            self.model.ent_coef = self.ENT_COEF_BOOST
            self._boosted = True
            print(f"\n[EntropyMonitor] Entropy collapsed ({actual_entropy:.3f}). Boosting ent_coef -> {self.ENT_COEF_BOOST}")
        elif actual_entropy >= self.ENTROPY_THRESHOLD and self._boosted:
            self.model.ent_coef = self.ENT_COEF_NORMAL
            self._boosted = False
            print(f"\n[EntropyMonitor] Entropy recovered ({actual_entropy:.3f}). Restoring ent_coef -> {self.ENT_COEF_NORMAL}")

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
        self.gen_data = GenData.from_gen(2)

        # Species encoding: use the built-in National Dex `num` field directly.
        # No hand-written list needed — GenData.pokedex already has every species.
        # _get_species_id() normalises num to [-1, 1] over the Gen 1+2 range (1–251).
        # (Stored on gen_data, which is already initialised above.)

        try:
            all_items = list(self.gen_data.items.keys())
        except AttributeError:
            all_items = []
        self._item_to_id = {name: i + 1 for i, name in enumerate(all_items)}
        self._item_count = max(len(self._item_to_id), 1)

        self._prev_opp_fainted: int = 0
        self._prev_my_fainted: int = 0
        self._consec_wasted_switches: int = 0
        self._last_action_was_switch: bool = False
        self._deadlock_turns: int = 0
        self._terminal_reward: Optional[float] = None
        self._battle_won: Optional[bool] = None

        # Cumulative counters — never reset, read directly by eval_model
        self.eval_wins: int = 0
        self.eval_losses: int = 0
        self.eval_draws: int = 0

        self._prev_active_opp_species: Optional[str] = None
        self._prev_active_own_species: Optional[str] = None
        self._prev_active_opp_hp: float = 1.0
        self._prev_active_own_hp: float = 1.0

    def action_to_order(self, action, battle, fake=False, strict=True):
        if action == 10:
            action = -2
        return super().action_to_order(action, battle, fake=fake, strict=strict)

    def _get_speed(self, mon) -> float:
        if mon is None or mon.species is None: return 0.0
        try:
            speed = self.gen_data.pokedex[mon.species.lower()]["baseStats"]["spe"]

            return (speed / MAX_SPEED) * 2 - 1
        except (KeyError, TypeError): return 0.0

    def _get_species_num(self, mon) -> float:
        """
        Returns the species National Dex number normalised to [-1, 1] over 1–251.
        Uses the `num` field from GenData.pokedex — no hand-written species list needed.
        Unknown species (not in pokedex, or num outside 1–251) return 0.0.
        """
        if mon is None or mon.species is None:
            return 0.0
        try:
            num = self.gen_data.pokedex[mon.species.lower()]["num"]
            if not (1 <= num <= 251):
                return 0.0
            return (num / SPECIES_NUM_SCALE) * 2 - 1
        except (KeyError, TypeError):
            return 0.0

    def _get_item_id(self, mon) -> float:
        if mon is None: return 0.0
        try:
            item = mon.item
            if item is None or item == "" or item == "unknown_item": return 0.0
            item_id = self._item_to_id.get(item.lower(), 0)
            if item_id == 0: return 0.0
            return (item_id / self._item_count) * 2 - 1
        except Exception: return 0.0

    def _get_type_advantage(self, attacker, defender) -> float:
        if attacker is None or defender is None: return 1.0
        best = 1.0
        for move in attacker.moves.values():
            try:
                mult = move.type.damage_multiplier(defender.type_1, defender.type_2, type_chart=self.gen_data.type_chart)
                if mult > best: best = mult
            except Exception: pass
        return best

    def _get_opp_move_effectiveness(self, battle) -> np.ndarray:
        features = np.zeros(4, dtype=np.float32)
        if battle.opponent_active_pokemon is None or battle.active_pokemon is None: return features
        for i, move in enumerate(list(battle.opponent_active_pokemon.moves.values())[:4]):
            try:
                eff = move.type.damage_multiplier(battle.active_pokemon.type_1, battle.active_pokemon.type_2, type_chart=self.gen_data.type_chart)
                features[i] = np.clip(eff - 1.0, -1.0, 1.0)
            except (AssertionError, TypeError): features[i] = 0.0
        return features

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any], env_id: int = 0) -> SingleAgentWrapper:
        agent_config = AccountConfiguration(f"agent_{env_id}", None)
        opponent_config = AccountConfiguration(f"Opponent_bot_{env_id}", None)
        env = cls(
            battle_format=config["battle_format"],
            log_level=30,
            # FIX: open_timeout=None can hang indefinitely if the Showdown server is slow.
            open_timeout=60,
            strict=False,
            account_configuration1=agent_config,
            account_configuration2=opponent_config,
            server_configuration=LocalhostServerConfiguration,
        )
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        opponent = RandomPlayer(start_listening=False, account_configuration=opponent_config)
        #opponent = MaxDamagePlayer(start_listening=False, account_configuration=opponent_config)
        #opponent = SimpleHeuristicsPlayer(start_listening=False, account_configuration=opponent_config)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        base_env = SingleAgentWrapper(env, opponent)
        return ActionMasker(base_env, mask_env)

    def _reset_battle_tracking(self):
        self._prev_opp_fainted = 0
        self._prev_my_fainted = 0
        self._consec_wasted_switches = 0
        self._last_action_was_switch = False
        self._deadlock_turns = 0
        self._terminal_reward = None
        self._battle_won = None
        self._prev_active_opp_species = None
        self._prev_active_own_species = None
        self._prev_active_opp_hp = 1.0
        self._prev_active_own_hp = 1.0

    def _get_team_hp_fraction(self, team) -> float:
        total_current = total_max = 0.0
        for mon in team.values():
            if not mon.fainted and mon.max_hp > 0:
                total_current += mon.current_hp
                total_max += mon.max_hp
        return 0.0 if total_max == 0 else total_current / total_max

    def calc_reward(self, battle) -> float:
        reward = 0.0

        # Normalize perspective: always compute from agent1's point of view.
        # battle2 is the opponent's mirror — team/opponent_team are swapped there.
        is_agent_battle = (battle is self.battle1)
        if is_agent_battle:
            my_team    = battle.team
            opp_team   = battle.opponent_team
            my_active  = battle.active_pokemon
            opp_active = battle.opponent_active_pokemon
        else:
            my_team    = battle.opponent_team
            opp_team   = battle.team
            my_active  = battle.opponent_active_pokemon
            opp_active = battle.active_pokemon

        if battle.finished:
            if battle.won is None:
                terminal = -0.5
            elif battle.won:
                terminal = 1.0
            else:
                terminal = -1.0
            if battle.won:
                terminal += max(0.0, (80 - battle.turn) / 80) * 0.15

            self._terminal_reward = terminal
            self._battle_won = battle.won

            if is_agent_battle:
                if battle.won is True:
                    self.eval_wins += 1
                elif battle.won is False:
                    self.eval_losses += 1
                else:
                    self.eval_draws += 1

            self._reset_battle_tracking()
            return terminal

        # --- KO delta ---
        opp_fainted_now = sum(p.fainted for p in opp_team.values())
        my_fainted_now  = sum(p.fainted for p in my_team.values())
        reward += 0.08 * (opp_fainted_now - self._prev_opp_fainted)
        reward -= 0.08 * (my_fainted_now  - self._prev_my_fainted)
        self._prev_opp_fainted = opp_fainted_now
        self._prev_my_fainted  = my_fainted_now

        # --- HP damage delta (per-active-mon, species-gated to avoid switch noise) ---
        HP_CHANGE_THRESHOLD = 0.015
        opp_species = opp_active.species if opp_active else None
        own_species  = my_active.species  if my_active  else None

        opp_hp_now = (opp_active.current_hp / opp_active.max_hp
                    if opp_active and not opp_active.fainted and opp_active.max_hp > 0 else 0.0)
        own_hp_now  = (my_active.current_hp  / my_active.max_hp
                    if my_active  and not my_active.fainted  and my_active.max_hp  > 0 else 0.0)

        if opp_species == self._prev_active_opp_species and self._prev_active_opp_species is not None:
            opp_hp_lost = max(0.0, self._prev_active_opp_hp - opp_hp_now)
            opp_hp_actually_dropped = opp_hp_lost > HP_CHANGE_THRESHOLD
            reward += 0.015 * opp_hp_lost
        else:
            opp_hp_lost = 0.0
            opp_hp_actually_dropped = False

        if own_species == self._prev_active_own_species and self._prev_active_own_species is not None:
            own_hp_lost = max(0.0, self._prev_active_own_hp - own_hp_now)
            own_hp_actually_dropped = own_hp_lost > HP_CHANGE_THRESHOLD
            reward -= 0.01 * own_hp_lost
        else:
            own_hp_lost = 0.0
            own_hp_actually_dropped = False

        self._prev_active_opp_species = opp_species
        self._prev_active_own_species  = own_species
        self._prev_active_opp_hp      = opp_hp_now
        self._prev_active_own_hp       = own_hp_now

        # --- Small flat time penalty ---
        reward -= 0.003

        # Late-game time pressure.
        if battle.turn > 40:
            reward -= min((battle.turn - 60) * 0.001, 0.05)

        # --- Deadlock penalty ---
        if not opp_hp_actually_dropped and not own_hp_actually_dropped and not battle.force_switch:
            self._deadlock_turns += 1
            if self._deadlock_turns > 5:
                reward -= min(0.005 * (self._deadlock_turns - 5), 0.05)
        else:
            self._deadlock_turns = 0

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
                            ) - 1.0, -1.0, 1.0)
                    moves_pp_ratio[i] = (move.current_pp / move.max_pp) * 2 - 1
                except AssertionError: pass
            # Species identifiers — National Dex num from GenData, normalised to [-1, 1]
            for i, (_, mon) in enumerate(sorted(battle.team.items())):
                team_identifier[i] = self._get_species_num(mon)
            for i, (_, mon) in enumerate(sorted(battle.opponent_team.items())):
                opponent_identifier[i] = self._get_species_num(mon)

            # FIX: Use mon.status.name.lower() instead of mon.status.value.
            # In poke-env, Status is an IntEnum so .value returns an integer (e.g. 1),
            # which never matches the string keys in STATUS_MAP ("brn", "slp", etc.).
            # This bug caused all status lookups to silently return 0.0 (no status).
            for i, mon in enumerate(battle.team.values()):
                status_key = mon.status.name.lower() if mon.status else None
                self_status[i] = STATUS_MAP.get(status_key, 0.0)
            for i, mon in enumerate(battle.opponent_team.values()):
                status_key = mon.status.name.lower() if mon.status else None
                opponent_status[i] = STATUS_MAP.get(status_key, 0.0)

            if len(battle.available_moves) == 0 and len(battle.available_switches) == 0:
                special_case[0] = 1
            if battle.active_pokemon.must_recharge:
                special_case[1] = 1

            # _encode_active_mon returns 37 dims [hp, type1_onehot(18), type2_onehot(18)]
            active_features     = _encode_active_mon(battle.active_pokemon)
            opp_active_features = _encode_active_mon(battle.opponent_active_pokemon)
            own_speed    = np.float32([self._get_speed(battle.active_pokemon)])
            opp_speed    = np.float32([self._get_speed(battle.opponent_active_pokemon)])
            opp_move_eff = self._get_opp_move_effectiveness(battle)
            own_boosts   = _encode_boosts(battle.active_pokemon)
            opp_boosts   = _encode_boosts(battle.opponent_active_pokemon)
            own_item_id  = np.float32([self._get_item_id(battle.active_pokemon)])
            opp_item_id  = np.float32([self._get_item_id(battle.opponent_active_pokemon)])
            own_status_flags = _encode_status_flags(battle.active_pokemon)
            opp_status_flags = _encode_status_flags(battle.opponent_active_pokemon)
            turn_counter = np.float32([min(battle.turn / 150.0, 1.0)])

            # Explicit binary alive flags — cleaner than relying on the network to decode
            # the -1.0 HP sentinel in team_hp_ratio as meaning "fainted".
            own_alive_flags = np.float32([0.0 if mon.fainted else 1.0
                                          for mon in battle.team.values()]
                                         + [1.0] * (6 - len(battle.team)))
            opp_alive_flags = np.float32([0.0 if mon.fainted else 1.0
                                          for mon in battle.opponent_team.values()]
                                         + [1.0] * (6 - len(battle.opponent_team)))

            return np.float32(np.concatenate([
                moves_base_power,        # 4
                moves_dmg_multiplier,    # 4
                moves_pp_ratio,          # 4
                team_hp_ratio,           # 6
                opponent_hp_ratio,       # 6
                team_identifier,         # 6
                opponent_identifier,     # 6
                self_status,             # 6
                opponent_status,         # 6
                special_case,            # 2
                active_features,         # 37
                opp_active_features,     # 37
                own_speed,               # 1
                opp_speed,               # 1
                opp_move_eff,            # 4
                own_boosts,              # 5
                opp_boosts,              # 5
                own_item_id,             # 1
                opp_item_id,             # 1
                own_status_flags,        # 5
                opp_status_flags,        # 5
                turn_counter,            # 1
                own_alive_flags,         # 6
                opp_alive_flags,         # 6
            ]))
            # Total: 4+4+4+6+6+6+6+6+6+2+37+37+1+1+4+5+5+1+1+5+5+1+6+6 = 165

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
        self._last_action_was_switch = raw_action in range(0, 6)
        self._terminal_reward = None
        self._battle_won = None
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            if self._terminal_reward is not None:
                info["terminal_reward"] = self._terminal_reward
            if self._battle_won is not None:
                info["battle_won"] = self._battle_won
        return obs, reward, terminated, truncated, info


# -----------------------------
# Wrapper chain walker
# -----------------------------
def _get_custom_env(env) -> Optional[CustomEnv]:
    obj = env
    while obj is not None:
        if isinstance(obj, CustomEnv): return obj
        obj = getattr(obj, 'env', None)
    return None


# -----------------------------
# Move allow helper (used by mask_env)
# -----------------------------
def _is_move_allowed(move, own_mon, battle, boosts, opp_remaining, own_incapacitated) -> bool:
    # Sleep Talk only usable while asleep
    if move.id == "sleeptalk" and not _is_asleep(own_mon):
        return False

    # Rest: blocked if already asleep, HP too high, or healthy with no status
    if move.id == "rest":
        hp_ratio = own_mon.current_hp / own_mon.max_hp if own_mon.max_hp > 0 else 0
        if _is_asleep(own_mon):
            return False
        if hp_ratio > 0.8:
            return False
        if own_mon.status is None and hp_ratio > 0.5:
            return False

    # Other healing moves: block at high HP
    if getattr(move, 'heal', 0) and move.id != "rest":
        hp_ratio = own_mon.current_hp / own_mon.max_hp if own_mon.max_hp > 0 else 0
        if hp_ratio > 0.85:
            return False

    # Status moves: block if opponent already has a status condition
    if getattr(move, 'status', None) and battle.opponent_active_pokemon is not None:
        if battle.opponent_active_pokemon.status is not None:
            return False

    # Stat boost moves: block if the relevant stat is already maxed (+6)
    if move.base_power == 0 or move.base_power is None:
        move_id = move.id
        if move_id in ("agility", "rocksmash") and boosts.get("spe", 0) >= 6:
            return False
        if move_id in ("swordsdance", "meditate", "sharpen") and boosts.get("atk", 0) >= 6:
            return False
        if move_id == "growth" and boosts.get("spa", 0) >= 6 and boosts.get("spd", 0) >= 6:
            return False
        if move_id in ("nastyplot", "chargebeam") and boosts.get("spa", 0) >= 6:
            return False
        if move_id in ("defensecurl", "harden", "withdraw") and boosts.get("def", 0) >= 6:
            return False
        if move_id == "curse" and boosts.get("atk", 0) >= 6 and boosts.get("def", 0) >= 6:
            return False

    # Block zero-power moves entirely when incapacitated (asleep or frozen),
    # except Sleep Talk which is the only valid action while asleep.
    if own_incapacitated and (move.base_power == 0 or move.base_power is None):
        if move.id != "sleeptalk":
            return False

    return True


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

    own_mon = battle.active_pokemon
    own_incapacitated = _is_asleep(own_mon) or _is_frozen(own_mon)
    boosts = own_mon.boosts if own_mon.boosts is not None else {}
    opp_remaining = sum(1 for m in battle.opponent_team.values() if not m.fainted)

    # --- Build move mask ---
    if not battle.force_switch or not battle.active_pokemon.fainted:
        for slot, move in enumerate(moves):
            if move not in available_moves:
                continue
            if _is_move_allowed(move, own_mon, battle, boosts, opp_remaining, own_incapacitated):
                action_mask[slot + move_offset] = 1

    # --- Build switch mask ---
    allow_switch = battle.force_switch or own_incapacitated or len(available_moves) > 0
    if allow_switch:
        for slot, mon in enumerate(team):
            if mon in available_switches:
                action_mask[slot] = 1

    if not battle.force_switch:
        move_slots_open = any(action_mask[move_offset:move_offset + len(moves)])
        if not move_slots_open and any(action_mask[:6]):
            pass  # switches only — already correct

    if not any(action_mask):
        action_mask[choose_default] = 1

    return action_mask


# -----------------------------
# Learning rate schedule
# -----------------------------
def linear_lr_schedule(initial_lr: float):
    """
    Returns a callable that linearly decays the learning rate from
    initial_lr to 0 as training progress goes from 1.0 to 0.0.
    Pass this as learning_rate= when constructing the PPO model.
    """
    def schedule(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return schedule


# -----------------------------
# Parallel env factory
# -----------------------------
def make_env_fn(env_id: int, seed: int = 0, battle_format: str = "gen2randombattle"):
    def _init():
        set_random_seed(seed + env_id)
        env = CustomEnv.create_single_agent_env({"battle_format": battle_format}, env_id=env_id)
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