import asyncio
import logging
import random
import string
import time
import numpy as np
from typing import Any, Dict, Optional

# Per-process random suffix, generated once when this module is imported.
# Showdown keeps unfinished battles alive for unregistered guest usernames for a few
# minutes after disconnect — if a previous run crashed, reusing usernames like
# "agent_0" causes the new run to re-inherit those zombie battles, which then makes
# poke-env's reset_battles() raise "Can not reset player's battles while they are
# still running". A 4-char random suffix per run guarantees fresh usernames.
# SubprocVecEnv with start_method="spawn" re-imports this module per child, so each
# subprocess gets its own suffix — agent and opponent within the same subprocess
# share it (they both come from the same import) and so still pair correctly.
_RUN_SUFFIX = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))

from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.battle import AbstractBattle, Battle, Weather, SideCondition
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
    name_str = str(name) if name else ""
    # Match both legacy ("agent_0", "Opponent_bot_0") and new ("agt_0_abcd", "opp_0_abcd")
    # username schemes so log filtering survives the per-run-suffix rename.
    if name_str.startswith(("opp_", "Opponent_bot")):
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    if name_str.startswith(("agt_", "agent")):
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

def _patched_order_to_action(order, battle, fake=False, strict=True):
    # Must be a staticmethod so instance calls (self.env.order_to_action(...)) do not
    # prepend the env instance as a spurious first argument — which would cause
    # AssertionError inside the original, silently caught, and return 10 every time.
    try:
        return _original_order_to_action(order, battle, fake, strict)
    except (ValueError, RecursionError, AssertionError):
        return 10

_singles_env_module.SinglesEnv.order_to_action = staticmethod(_patched_order_to_action)

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
    # Threshold lowered to only fire on real collapse — previously 1.0 sat above the
    # natural equilibrium entropy (~0.95–1.0), so the boost was permanently on, which
    # pinned the policy near uniform and prevented commitment to learned moves.
    ENTROPY_THRESHOLD = 0.3
    ENT_COEF_BOOST = 0.03
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

        self._last_move_action: int = -1
        self._consec_same_move: int = 0

        # Positive stat-boost sum of own active mon at end of last calc_reward call —
        # used to detect "wasted setup" (e.g. Curse then immediate switch).
        self._prev_own_pos_boost_sum: float = 0.0
        # True if own active species changed during last calc_reward (forced or voluntary).
        # Catches double-switch waste like forced-switch-on-KO followed by voluntary switch.
        self._switched_last_turn: bool = False
        # True ONLY if last turn was a VOLUNTARY switch (not a forced KO-switch).
        # Used by the consecutive-switch penalty so it doesn't fire on the
        # standard "lose mon → forced send-out → switch to counter" play pattern.
        self._voluntary_switched_last_turn: bool = False

        # Per-species HP memory — credits damage continuously across switches.
        # Without this, switching after taking damage zeros the species-gated own_hp_lost
        # signal, letting the agent dodge the negative-HP penalty by spam-switching.
        self._opp_hp_by_species: Dict[str, float] = {}
        self._own_hp_by_species: Dict[str, float] = {}

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
    def create_single_agent_env(cls, config: Dict[str, Any], env_id: int = 0, use_opponent_cycle: bool = True) -> SingleAgentWrapper:
        # Per-run unique usernames — see _RUN_SUFFIX comment near the top of this module.
        # "agt_<id>_<suffix>" / "opp_<id>_<suffix>" stays well under Showdown's 18-char limit.
        agent_config = AccountConfiguration(f"agt_{env_id}_{_RUN_SUFFIX}", None)
        opponent_config = AccountConfiguration(f"opp_{env_id}_{_RUN_SUFFIX}", None)
        env = cls(
            battle_format=config["battle_format"],
            log_level=30,
            # FIX: open_timeout=None can hang indefinitely if the Showdown server is slow.
            open_timeout=60,
            # Raised from default 60s — with 6 parallel envs racing to challenge each
            # other on startup, the Showdown server can be slow enough that the agent
            # doesn't fill its battle queue within 60s, raising "Agent is not challenging".
            challenge_timeout=180.0,
            strict=False,
            account_configuration1=agent_config,
            account_configuration2=opponent_config,
            server_configuration=LocalhostServerConfiguration,
        )
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Opponent selection: use_opponent_cycle controls the strategy.
        # - use_opponent_cycle=False (new models): all envs use RandomPlayer for
        #   broad exploration without specialising. Quick baseline.
        # - use_opponent_cycle=True (continue training): cycle Heuristics/MaxDamage
        #   across envs so every rollout sees both strong opponents. Prevents
        #   catastrophic forgetting (Agent13 symptom: high approx_kl + falling ep_rew_mean).
        if use_opponent_cycle:
            opponent_classes = [SimpleHeuristicsPlayer, MaxDamagePlayer]
            OpponentClass = opponent_classes[env_id % len(opponent_classes)]
        else:
            OpponentClass = RandomPlayer
        opponent = OpponentClass(start_listening=False, account_configuration=opponent_config)
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
        self._opp_hp_by_species = {}
        self._own_hp_by_species = {}
        self._last_move_action = -1
        self._consec_same_move = 0
        self._prev_own_pos_boost_sum = 0.0
        self._switched_last_turn = False
        self._voluntary_switched_last_turn = False

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
            # Terminal reward boosted (was ±1.0) so winning dominates the gradient.
            # Prior balance let the policy farm shaping rewards while losing battles
            # — reward could trend positive while winrate stagnated.
            if battle.won is None:
                terminal = -1.0
            elif battle.won:
                terminal = 2.0
            else:
                terminal = -2.0
            if battle.won:
                terminal += max(0.0, (80 - battle.turn) / 80) * 0.3

            # BUG FIX: gate _terminal_reward / _battle_won to is_agent_battle.
            # calc_reward is called for BOTH battles per step. battle.won is
            # perspective-dependent (battle1.won == True iff agent won;
            # battle2.won == True iff opponent won), so the second call would
            # overwrite the agent's terminal with the opponent's, making
            # ProgressCallback report wins as losses under SubprocVecEnv.
            if is_agent_battle:
                self._terminal_reward = terminal
                self._battle_won = battle.won

                if battle.won is True:
                    self.eval_wins += 1
                elif battle.won is False:
                    self.eval_losses += 1
                else:
                    self.eval_draws += 1

                self._reset_battle_tracking()
            return terminal

        # --- KO delta ---
        # Agent14 rebalance: Agent13 had own_hp_lost (0.05) > opp_hp_lost (0.04), so
        # every even HP trade was net-negative reward. The policy learned to avoid
        # combat (Misdreavus-into-Murkrow swap, Skarmory eating fatal Surf instead
        # of pivoting). Restored to opp_hp_lost >= own_hp_lost so successful trades
        # are rewarded; HP preservation still matters but no longer dominates.
        opp_fainted_now = sum(p.fainted for p in opp_team.values())
        my_fainted_now  = sum(p.fainted for p in my_team.values())
        reward += 0.10 * (opp_fainted_now - self._prev_opp_fainted)
        reward -= 0.10 * (my_fainted_now  - self._prev_my_fainted)
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

        # Use per-species HP memory so switching out then back in still tracks
        # damage dealt/taken on that mon. Agent19: only credit damage if we've
        # already seen the species — first-sight defaults to recording current HP
        # without crediting anything. Prior code defaulted prev=1.0, so an opp
        # mon switched in already at <100% (Sandstorm/Spikes/Leftovers prior turn)
        # gave a phantom +0.04 * (1 - hp_now) reward the agent didn't earn.
        if opp_species is not None:
            if opp_species in self._opp_hp_by_species:
                prev_opp_hp = self._opp_hp_by_species[opp_species]
                opp_hp_lost = max(0.0, prev_opp_hp - opp_hp_now)
                opp_hp_actually_dropped = opp_hp_lost > HP_CHANGE_THRESHOLD
                reward += 0.04 * opp_hp_lost
            else:
                opp_hp_lost = 0.0
                opp_hp_actually_dropped = False
            self._opp_hp_by_species[opp_species] = opp_hp_now
        else:
            opp_hp_lost = 0.0
            opp_hp_actually_dropped = False

        if own_species is not None:
            if own_species in self._own_hp_by_species:
                prev_own_hp = self._own_hp_by_species[own_species]
                own_hp_lost = max(0.0, prev_own_hp - own_hp_now)
                own_hp_actually_dropped = own_hp_lost > HP_CHANGE_THRESHOLD
                # Agent14: was 0.05 (loss-averse, exceeded opp_hp_lost). Now matches
                # opp_hp_lost (0.04) so even-trades are net-zero, not net-negative.
                reward -= 0.04 * own_hp_lost
            else:
                own_hp_lost = 0.0
                own_hp_actually_dropped = False
            self._own_hp_by_species[own_species] = own_hp_now
        else:
            own_hp_lost = 0.0
            own_hp_actually_dropped = False

        # --- Switch-pattern penalties (gated to agent perspective so per-step state
        # mutations happen exactly once even if calc_reward is called for both battles) ---
        if is_agent_battle:
            own_species_changed = (
                own_species is not None
                and self._prev_active_own_species is not None
                and own_species != self._prev_active_own_species
            )
            # Voluntary switch requires:
            #  1. Active species actually changed
            #  2. The agent's chosen action this turn was a switch slot (0-5).
            #     Filters out Whirlwind/Roar phazes and Baton Pass — both change species
            #     without the agent choosing a switch action, so they should not be penalised.
            #  3. The previous active is still alive (otherwise it's a forced switch on KO).
            voluntary_switch = False
            if own_species_changed and self._last_action_was_switch:
                prev_mon = next(
                    (m for m in my_team.values()
                     if m.species == self._prev_active_own_species),
                    None,
                )
                if prev_mon is not None and not prev_mon.fainted:
                    voluntary_switch = True

            # Boost retention: voluntary switch abandons stat boosts (Pokemon resets
            # boosts on switch). Penalises wasted setup like Curse → immediate switch.
            # Kept in Agent18 — plain game mechanics, not learnable shaping, can't be farmed.
            if voluntary_switch and self._prev_own_pos_boost_sum > 0:
                reward -= 0.04 * self._prev_own_pos_boost_sum



            # Track voluntary-only switches separately so the consecutive-switch
            # penalty above doesn't fire after forced-on-KO switches.
            self._voluntary_switched_last_turn = voluntary_switch
            self._switched_last_turn = own_species_changed and self._last_action_was_switch
            if my_active is not None and my_active.boosts:
                self._prev_own_pos_boost_sum = sum(
                    max(0, v) for v in my_active.boosts.values()
                )
            else:
                self._prev_own_pos_boost_sum = 0.0

        self._prev_active_opp_species = opp_species
        self._prev_active_own_species  = own_species
        self._prev_active_opp_hp      = opp_hp_now
        self._prev_active_own_hp       = own_hp_now


        return reward

    def embed_battle(self, battle: AbstractBattle):
        # Delegate to the canonical free-function implementation. InferenceAgent._embed
        # delegates to the same function, guaranteeing byte-identical observations
        # for training (P1) and inference (P2).
        return embed_battle_impl(battle, self.gen_data)

    def reset(self, *args, **kwargs):
        self._reset_battle_tracking()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return super().reset(*args, **kwargs)
            except (asyncio.TimeoutError, OSError) as e:
                # On a timeout, poke-env's super().reset() left a challenge task running
                # in POKE_LOOP and may have populated agent._battles partway. The next
                # attempt would either issue a duplicate challenge or hit
                # "Can not reset player's battles while they are still running". Cancel
                # the dangling task and clear stale state before retrying.
                if attempt < max_retries - 1:
                    logging.warning(
                        f"[Env] reset failed (attempt {attempt + 1}/{max_retries}): "
                        f"{type(e).__name__}: {e}. Cleaning up and retrying..."
                    )
                    challenge_task = getattr(self, "_challenge_task", None)
                    if challenge_task is not None:
                        try:
                            challenge_task.cancel()
                        except Exception:
                            pass
                        self._challenge_task = None
                    # Drop any half-formed battle state on both player objects so the
                    # next reset_battles() sees a clean slate.
                    for player in (getattr(self, "agent1", None), getattr(self, "agent2", None)):
                        if player is not None:
                            try:
                                player._battles = {}
                            except Exception:
                                pass
                    time.sleep(3)
                else:
                    raise

    def close(self):
        super().close()
        print("[Environment Closed]")

    def step(self, action):
        if isinstance(action, dict):
            raw_action = next(iter(action.values()))
        else:
            raw_action = action
        self._last_action_was_switch = raw_action in range(0, 6)
        # Track move repetition (move slots 6-9; ignore switches and default action 10)
        if 6 <= raw_action <= 9:
            if raw_action == self._last_move_action:
                self._consec_same_move += 1
            else:
                self._consec_same_move = 0
                self._last_move_action = raw_action
        else:
            self._consec_same_move = 0
            self._last_move_action = -1
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
def _is_move_allowed(move, own_mon, battle, boosts, own_incapacitated) -> bool:
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

    # Weather moves: block if the same weather is already active
    if getattr(move, 'weather', None) is not None and move.weather in battle.weather:
        return False

    # Curse: block if ATK and DEF are both maxed (checked outside base_power gate for robustness)
    if move.id == "curse" and boosts.get("atk", 0) >= 6 and boosts.get("def", 0) >= 6:
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

    # Block zero-power moves entirely when incapacitated (asleep or frozen),
    # except Sleep Talk which is the only valid action while asleep.
    if own_incapacitated and (move.base_power == 0 or move.base_power is None):
        if move.id != "sleeptalk":
            return False

    return True


# -----------------------------
# Shared embedding helpers (free functions — used by BOTH training (CustomEnv.embed_battle)
# and inference (InferenceAgent._embed) to guarantee byte-identical observations.
# Do NOT add a parallel implementation; always call embed_battle_impl from both paths.
# -----------------------------
def _get_speed_for(mon, gen_data) -> float:
    if mon is None or mon.species is None:
        return 0.0
    try:
        speed = gen_data.pokedex[mon.species.lower()]["baseStats"]["spe"]
        return (speed / MAX_SPEED) * 2 - 1
    except (KeyError, TypeError):
        return 0.0


def _get_species_num_for(mon, gen_data) -> float:
    if mon is None or mon.species is None:
        return 0.0
    try:
        num = gen_data.pokedex[mon.species.lower()]["num"]
        if not (1 <= num <= 251):
            return 0.0
        return (num / SPECIES_NUM_SCALE) * 2 - 1
    except (KeyError, TypeError):
        return 0.0


def _get_opp_move_effectiveness_for(battle, gen_data) -> np.ndarray:
    features = np.zeros(4, dtype=np.float32)
    if battle.opponent_active_pokemon is None or battle.active_pokemon is None:
        return features
    for i, move in enumerate(list(battle.opponent_active_pokemon.moves.values())[:4]):
        try:
            eff = move.type.damage_multiplier(
                battle.active_pokemon.type_1, battle.active_pokemon.type_2,
                type_chart=gen_data.type_chart)
            features[i] = np.clip(eff - 1.0, -1.0, 1.0)
        except (AssertionError, TypeError):
            features[i] = 0.0
    return features


def _best_off_multiplier(mon, opp, gen_data) -> float:
    """
    Best raw damage multiplier among `mon`'s damaging moves vs `opp`'s type pair.
    Returns 1.0 when there is no info (no opp / no mon / no damaging moves seen).
    Used by both the obs builder (bench_off_multiplier feature) and the reward
    function (switch-into-better-matchup bonus) so the two stay consistent.
    """
    if mon is None or opp is None:
        return 1.0
    best = 0.0
    has_attack = False
    for move in mon.moves.values():
        if move.base_power is None or move.base_power == 0:
            continue
        try:
            mult = move.type.damage_multiplier(
                opp.type_1, opp.type_2, type_chart=gen_data.type_chart
            )
            has_attack = True
            if mult > best:
                best = mult
        except (AssertionError, TypeError):
            pass
    return best if has_attack else 1.0


def _worst_def_multiplier(mon, opp, gen_data) -> float:
    """
    Worst (highest) damage multiplier among `opp`'s *seen* damaging moves vs
    `mon`'s type pair. Returns 1.0 when no info. Mirrors the per-team
    bench_def_vulnerability feature — used by the defensive-switch reward
    so reward and obs share the same matchup view.
    """
    if mon is None or opp is None:
        return 1.0
    worst = 0.0
    seen_any = False
    for move in opp.moves.values():
        if move.base_power is None or move.base_power == 0:
            continue
        try:
            mult = move.type.damage_multiplier(
                mon.type_1, mon.type_2, type_chart=gen_data.type_chart
            )
            seen_any = True
            if mult > worst:
                worst = mult
        except (AssertionError, TypeError):
            pass
    return worst if seen_any else 1.0


def _get_bench_off_multiplier_for(battle, gen_data) -> np.ndarray:
    """
    For each of 6 team slots (insertion order, same as team_hp_ratio etc.), the best
    damage multiplier among that mon's damaging moves vs the opponent's active type.
    Encoded as clip(best_mult - 1.0, -1.0, 1.0). Fainted mons → 0.0.
    Tells the policy "switching to slot k gives a super-effective option."
    """
    features = np.zeros(6, dtype=np.float32)
    opp = battle.opponent_active_pokemon
    if opp is None:
        return features
    for i, mon in enumerate(battle.team.values()):
        if i >= 6 or mon.fainted:
            continue
        mult = _best_off_multiplier(mon, opp, gen_data)
        features[i] = np.clip(mult - 1.0, -1.0, 1.0)
    return features


def _get_bench_def_vulnerability_for(battle, gen_data) -> np.ndarray:
    """
    For each of 6 team slots (insertion order), the WORST damage multiplier the
    opponent's *seen* moves achieve vs that mon's type pair. Encoded as
    clip(worst_mult - 1.0, -1.0, 1.0). Fainted mons → 0.0. Partial info: only
    seen opponent moves are considered (same convention as opp_move_eff).
    Tells the policy "switching to slot k will get OHKO'd by a known threat."
    """
    features = np.zeros(6, dtype=np.float32)
    opp = battle.opponent_active_pokemon
    if opp is None:
        return features
    opp_moves = [m for m in opp.moves.values()
                 if m.base_power is not None and m.base_power > 0]
    if not opp_moves:
        return features
    for i, mon in enumerate(battle.team.values()):
        if i >= 6 or mon.fainted:
            continue
        worst = 0.0
        seen_any = False
        for move in opp_moves:
            try:
                mult = move.type.damage_multiplier(
                    mon.type_1, mon.type_2, type_chart=gen_data.type_chart
                )
                seen_any = True
                if mult > worst:
                    worst = mult
            except (AssertionError, TypeError):
                pass
        if seen_any:
            features[i] = np.clip(worst - 1.0, -1.0, 1.0)
    return features


def _encode_slot_features(mon, gen_data) -> np.ndarray:
    """
    42-dim per-team-slot rich feature vector:
      type1_onehot 18, type2_onehot 18, base_stats 6
        (hp/atk/def/spa/spd/spe normalised to [-1, 1] over 0-255).
    Returns zeros for None / unknown species — combine with the per-slot seen /
    alive flags so the network can detect "no info here" instead of treating
    a zero vector as a real mon.
    """
    if mon is None or mon.species is None:
        return np.zeros(42, dtype=np.float32)
    type1 = _type_onehot(mon.type_1)
    type2 = _type_onehot(mon.type_2)
    stats = np.zeros(6, dtype=np.float32)
    try:
        base = gen_data.pokedex[mon.species.lower()]["baseStats"]
        stats[0] = (base.get("hp",  0) / 255.0) * 2 - 1
        stats[1] = (base.get("atk", 0) / 255.0) * 2 - 1
        stats[2] = (base.get("def", 0) / 255.0) * 2 - 1
        stats[3] = (base.get("spa", 0) / 255.0) * 2 - 1
        stats[4] = (base.get("spd", 0) / 255.0) * 2 - 1
        stats[5] = (base.get("spe", 0) / 255.0) * 2 - 1
    except (KeyError, TypeError):
        pass
    return np.concatenate([type1, type2, stats])


def _get_team_slot_features(battle, gen_data) -> np.ndarray:
    """
    252-dim flat vector: 6 own-team slots × 42 dims, insertion order so slot k
    here corresponds to switch action k. Earlier obs only carried a single
    species_num scalar per slot — agents below Agent16 had to derive every
    bench mon's role from one nearly-arbitrary number on the dex axis, which
    capped policy quality at the heuristic-tier (~65% / 55% WR). Per-slot
    types + base stats give the policy actual identity for every bench mon.
    """
    features = np.zeros(6 * 42, dtype=np.float32)
    for i, mon in enumerate(list(battle.team.values())[:6]):
        features[i * 42:(i + 1) * 42] = _encode_slot_features(mon, gen_data)
    return features


def _get_opp_slot_features(battle, gen_data) -> np.ndarray:
    """
    252-dim flat vector for the opponent team in DISCOVERY order. Unseen slots
    are zero — pair with opp_seen_flags so the policy can distinguish
    "nothing here yet" from a real seen mon with all-zero stats.
    """
    features = np.zeros(6 * 42, dtype=np.float32)
    for i, mon in enumerate(list(battle.opponent_team.values())[:6]):
        features[i * 42:(i + 1) * 42] = _encode_slot_features(mon, gen_data)
    return features


def _get_opp_seen_flags(battle) -> np.ndarray:
    """
    1.0 for slots that hold a seen opp mon, 0.0 for slots not yet revealed.
    Earlier agents padded opp_alive_flags with 1.0 for unseen slots, which made
    "5 unseen opps" indistinguishable from "5 healthy seen opps" — endgame mon
    counting was effectively blind. opp_seen_flags closes that gap and
    opp_alive_flags is now 1.0 only for SEEN-AND-ALIVE.
    """
    flags = np.zeros(6, dtype=np.float32)
    n_seen = min(len(battle.opponent_team), 6)
    flags[:n_seen] = 1.0
    return flags


def _get_own_is_active(battle) -> np.ndarray:
    """6-dim one-hot: which own-team slot currently holds the active mon."""
    vec = np.zeros(6, dtype=np.float32)
    active = battle.active_pokemon
    if active is None:
        return vec
    for i, mon in enumerate(list(battle.team.values())[:6]):
        if mon is active:
            vec[i] = 1.0
            break
    return vec


def _get_opp_is_active(battle) -> np.ndarray:
    """6-dim one-hot: which seen-opp slot currently holds the opp active mon."""
    vec = np.zeros(6, dtype=np.float32)
    active = battle.opponent_active_pokemon
    if active is None:
        return vec
    for i, mon in enumerate(list(battle.opponent_team.values())[:6]):
        if mon is active:
            vec[i] = 1.0
            break
    return vec


def _boost_mult(b: int) -> float:
    """Stat-boost speed multiplier — Gen-2 standard: +b → (2+b)/2, -b → 2/(2-b)."""
    if b >= 0:
        return (2 + b) / 2.0
    return 2.0 / (2 - b)


def _get_speed_advantage(battle, gen_data) -> np.ndarray:
    """
    Single scalar in {+1, 0, -1} for who outspeeds, accounting for stat boosts
    and paralysis (which quarters speed in Gen 2). Saves the value head from
    deriving "I move first" out of base speeds + boosts + status flags every
    step — a strong signal for switch / status / setup decisions.
    """
    own = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    if own is None or opp is None or own.species is None or opp.species is None:
        return np.float32([0.0])
    try:
        own_base = gen_data.pokedex[own.species.lower()]["baseStats"]["spe"]
    except (KeyError, TypeError):
        own_base = 0
    try:
        opp_base = gen_data.pokedex[opp.species.lower()]["baseStats"]["spe"]
    except (KeyError, TypeError):
        opp_base = 0
    own_boost = own.boosts.get("spe", 0) if own.boosts else 0
    opp_boost = opp.boosts.get("spe", 0) if opp.boosts else 0
    own_eff = own_base * _boost_mult(own_boost)
    opp_eff = opp_base * _boost_mult(opp_boost)
    if _is_paralyzed(own):
        own_eff *= 0.25
    if _is_paralyzed(opp):
        opp_eff *= 0.25
    if own_eff > opp_eff:
        return np.float32([1.0])
    if own_eff < opp_eff:
        return np.float32([-1.0])
    return np.float32([0.0])


def _get_matchup_features_for(battle, gen_data) -> np.ndarray:
    """
    Three-scalar summary of the current active-vs-active matchup. Without this,
    Agent12 had to derive matchup quality from raw type one-hots every step —
    which produced the no-switch behaviour seen in battle logs (Quagsire staying
    in vs Kingdra Surf, Mewtwo eating two Earthquakes from Gligar at low HP).

    Returns:
      [own_off_advantage, opp_off_advantage, matchup_diff]
        own_off_advantage : clip(my_best_off_mult  - 1.0, -1, 1)
        opp_off_advantage : clip(opp_best_off_mult - 1.0, -1, 1)
        matchup_diff      : clip((own - opp) / 2.0, -1, 1)   (>0 = we are favoured)
    """
    own = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    if own is None or opp is None:
        return np.zeros(3, dtype=np.float32)
    my_off  = _best_off_multiplier(own, opp, gen_data)
    opp_off = _best_off_multiplier(opp, own, gen_data)
    return np.float32([
        np.clip(my_off  - 1.0, -1.0, 1.0),
        np.clip(opp_off - 1.0, -1.0, 1.0),
        np.clip((my_off - opp_off) / 2.0, -1.0, 1.0),
    ])


# Weather encoding — only the four weathers that exist in Gen 2.
_WEATHER_TO_IDX = {
    Weather.SUNNYDAY:  0,
    Weather.RAINDANCE: 1,
    Weather.SANDSTORM: 2,
    Weather.HAIL:      3,
}


def _encode_weather_for(battle) -> np.ndarray:
    vec = np.zeros(4, dtype=np.float32)
    weather = getattr(battle, "weather", None)
    if not weather:
        return vec
    for w in weather:
        idx = _WEATHER_TO_IDX.get(w)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _encode_side_conditions_for(side_conditions) -> np.ndarray:
    """
    Three-binary flags: [spikes, reflect, light_screen]. Gen 2 has single-layer
    Spikes only; Reflect/Light Screen halve damage of their respective categories
    for 5 turns. All three change switching/attacking calculus materially.
    """
    vec = np.zeros(3, dtype=np.float32)
    if not side_conditions:
        return vec
    if SideCondition.SPIKES in side_conditions:
        vec[0] = 1.0
    if SideCondition.REFLECT in side_conditions:
        vec[1] = 1.0
    if SideCondition.LIGHT_SCREEN in side_conditions:
        vec[2] = 1.0
    return vec


def embed_battle_impl(battle, gen_data) -> np.ndarray:
    """
    THE canonical 712-dim observation builder. Used by BOTH training (CustomEnv.embed_battle)
    and inference (InferenceAgent._embed). Any change here applies to both paths automatically.

    Layout (712 dims, all own-team-indexed slots use insertion order to match the action layer;
    opp-team-indexed slots use discovery order):
      moves_base_power 4, moves_dmg_multiplier 4, moves_pp_ratio 4,
      team_hp_ratio 6, opponent_hp_ratio 6,
      team_identifier 6, opponent_identifier 6,
      self_status 6, opponent_status 6, special_case 2,
      active_features 37, opp_active_features 37,
      own_speed 1, opp_speed 1, opp_move_eff 4,
      own_boosts 5, opp_boosts 5,
      own_status_flags 5, opp_status_flags 5,
      turn_counter 1, own_alive_flags 6, opp_alive_flags 6,
      bench_off_multiplier 6, bench_def_vulnerability 6,
      matchup_features 3, weather 4,
      own_side_conditions 3, opp_side_conditions 3,
      own_slot_features 252, opp_slot_features 252,
      opp_seen_flags 6, own_is_active 6, opp_is_active 6,
      force_switch_flag 1, speed_advantage_flag 1
    """
    try:
        moves_n, pokemon_team = 4, 6
        moves_base_power     = -np.ones(moves_n)
        moves_dmg_multiplier = np.ones(moves_n)
        moves_pp_ratio       = np.zeros(moves_n)
        team_hp_ratio        = np.ones(pokemon_team)
        opponent_hp_ratio    = np.ones(pokemon_team)
        team_identifier      = np.zeros(pokemon_team, dtype=np.float32)
        opponent_identifier  = np.zeros(pokemon_team, dtype=np.float32)
        self_status          = np.zeros(pokemon_team, dtype=np.float32)
        opponent_status      = np.zeros(pokemon_team, dtype=np.float32)
        special_case         = np.zeros(2, dtype=np.float32)

        # Insertion order throughout — matches the action layer (mask + action_to_order).
        # Earlier code mixed sorted-by-key and insertion order across HP/status/alive_flags,
        # forcing the policy to untangle inconsistent slot mappings inside one obs vector.
        for i, mon in enumerate(battle.team.values()):
            team_hp_ratio[i] = -1.0 if (mon.fainted or mon.max_hp == 0) else (mon.current_hp / mon.max_hp) * 2 - 1
        for i, mon in enumerate(battle.opponent_team.values()):
            opponent_hp_ratio[i] = -1.0 if (mon.fainted or mon.max_hp == 0) else (mon.current_hp / mon.max_hp) * 2 - 1

        # Index moves by the active mon's slot order — the SAME order the action layer
        # (mask + action_to_order) uses. Iterating battle.available_moves instead causes
        # an index/action mismatch whenever any move is disabled or out of PP.
        if battle.active_pokemon is not None:
            own_moves = list(battle.active_pokemon.moves.values())[:moves_n]
            for i, move in enumerate(own_moves):
                try:
                    moves_base_power[i] = (move.base_power / 250) * 2 - 1 if move.base_power is not None else 0.0
                    if battle.opponent_active_pokemon is not None:
                        moves_dmg_multiplier[i] = np.clip(
                            move.type.damage_multiplier(
                                battle.opponent_active_pokemon.type_1,
                                battle.opponent_active_pokemon.type_2,
                                type_chart=gen_data.type_chart
                            ) - 1.0, -1.0, 1.0)
                    if move.max_pp > 0:
                        moves_pp_ratio[i] = (move.current_pp / move.max_pp) * 2 - 1
                except AssertionError:
                    pass

        for i, mon in enumerate(battle.team.values()):
            team_identifier[i] = _get_species_num_for(mon, gen_data)
        for i, mon in enumerate(battle.opponent_team.values()):
            opponent_identifier[i] = _get_species_num_for(mon, gen_data)

        for i, mon in enumerate(battle.team.values()):
            status_key = mon.status.name.lower() if mon.status else None
            self_status[i] = STATUS_MAP.get(status_key, 0.0)
        for i, mon in enumerate(battle.opponent_team.values()):
            status_key = mon.status.name.lower() if mon.status else None
            opponent_status[i] = STATUS_MAP.get(status_key, 0.0)

        if len(battle.available_moves) == 0 and len(battle.available_switches) == 0:
            special_case[0] = 1
        if battle.active_pokemon is not None and battle.active_pokemon.must_recharge:
            special_case[1] = 1

        active_features     = _encode_active_mon(battle.active_pokemon)
        opp_active_features = _encode_active_mon(battle.opponent_active_pokemon)
        own_speed    = np.float32([_get_speed_for(battle.active_pokemon, gen_data)])
        opp_speed    = np.float32([_get_speed_for(battle.opponent_active_pokemon, gen_data)])
        opp_move_eff = _get_opp_move_effectiveness_for(battle, gen_data)
        own_boosts   = _encode_boosts(battle.active_pokemon)
        opp_boosts   = _encode_boosts(battle.opponent_active_pokemon)
        own_status_flags = _encode_status_flags(battle.active_pokemon)
        opp_status_flags = _encode_status_flags(battle.opponent_active_pokemon)
        turn_counter = np.float32([min(battle.turn / 150.0, 1.0)])

        # own_alive_flags pads with 1.0 because we always know all 6 own mons —
        # short team would mean a malformed battle. opp_alive_flags now pads with
        # 0.0: prior agents couldn't tell "5 unseen opps" from "5 healthy seen
        # opps", which broke endgame mon-counting. opp_seen_flags below carries
        # the "have I seen this slot yet" signal so the policy can recover the
        # old "presumed alive" interpretation when needed.
        own_alive_flags = np.float32([0.0 if mon.fainted else 1.0
                                      for mon in battle.team.values()]
                                     + [1.0] * (6 - len(battle.team)))
        opp_alive_flags = np.zeros(6, dtype=np.float32)
        for i, mon in enumerate(list(battle.opponent_team.values())[:6]):
            opp_alive_flags[i] = 0.0 if mon.fainted else 1.0

        bench_off_multiplier    = _get_bench_off_multiplier_for(battle, gen_data)
        bench_def_vulnerability = _get_bench_def_vulnerability_for(battle, gen_data)
        matchup_features        = _get_matchup_features_for(battle, gen_data)
        weather                 = _encode_weather_for(battle)
        own_side_conditions     = _encode_side_conditions_for(getattr(battle, "side_conditions", {}))
        opp_side_conditions     = _encode_side_conditions_for(getattr(battle, "opponent_side_conditions", {}))

        # Agent16: per-slot rich features (types + base stats), seen flags,
        # is-active one-hots, force-switch flag, speed-advantage flag.
        own_slot_features  = _get_team_slot_features(battle, gen_data)
        opp_slot_features  = _get_opp_slot_features(battle, gen_data)
        opp_seen_flags     = _get_opp_seen_flags(battle)
        own_is_active      = _get_own_is_active(battle)
        opp_is_active      = _get_opp_is_active(battle)
        force_switch_flag  = np.float32([1.0 if getattr(battle, "force_switch", False) else 0.0])
        speed_advantage    = _get_speed_advantage(battle, gen_data)

        return np.float32(np.concatenate([
            moves_base_power, moves_dmg_multiplier, moves_pp_ratio,
            team_hp_ratio, opponent_hp_ratio,
            team_identifier, opponent_identifier,
            self_status, opponent_status, special_case,
            active_features, opp_active_features,
            own_speed, opp_speed, opp_move_eff,
            own_boosts, opp_boosts,
            own_status_flags, opp_status_flags,
            turn_counter, own_alive_flags, opp_alive_flags,
            bench_off_multiplier, bench_def_vulnerability,
            matchup_features, weather,
            own_side_conditions, opp_side_conditions,
            own_slot_features, opp_slot_features,
            opp_seen_flags, own_is_active, opp_is_active,
            force_switch_flag, speed_advantage,
        ]))
    except Exception as e:
        # Surface embedding failures — silent zero-vectors hid bugs and produced
        # blind policy decisions during training. Limit log spam with a counter.
        embed_battle_impl._error_count = getattr(embed_battle_impl, "_error_count", 0) + 1
        if embed_battle_impl._error_count <= 5 or embed_battle_impl._error_count % 100 == 0:
            logging.getLogger(__name__).warning(
                f"[embed_battle] error #{embed_battle_impl._error_count}: {type(e).__name__}: {e}"
            )
        return np.zeros(OBS_SIZE, dtype=np.float32)


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

    # --- Build move mask ---
    # Old condition was `not force_switch OR not active.fainted` — i.e. only
    # suppressed moves when BOTH force_switch AND active was fainted. Saved by
    # available_moves being empty during force_switch, but the boolean intent
    # was inverted. Correct rule: no moves when forced to switch.
    if not battle.force_switch:
        for slot, move in enumerate(moves):
            if move not in available_moves:
                continue
            if _is_move_allowed(move, own_mon, battle, boosts, own_incapacitated):
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
    Linearly decays LR from initial_lr to 0.5 * initial_lr as progress goes 1.0 -> 0.0.
    Floor was 0.05 — that drove end-of-run LR to ~5e-6, killing clip_fraction/KL and
    freezing the policy (Agent9 symptom). 0.5 keeps gradients meaningful throughout.
    """
    floor = initial_lr * 0.5
    def schedule(progress_remaining: float) -> float:
        return max(initial_lr * progress_remaining, floor)
    return schedule


# -----------------------------
# Parallel env factory
# -----------------------------
def make_env_fn(env_id: int, seed: int = 0, battle_format: str = "gen2randombattle", use_opponent_cycle: bool = True, opponent_env_id: int = -1):
    def _init():
        set_random_seed(seed + env_id)
        # Stagger startup to avoid the thundering-herd at the local Showdown server:
        # when N subprocesses all open websockets and challenge each other in the same
        # ~10ms, the handshake can stall long enough to exceed challenge_timeout. A
        # small per-env delay flattens the connection spike.
        time.sleep(env_id * 2.0)
        # opponent_env_id overrides env_id for opponent selection (used by eval to
        # pin a specific opponent without changing the env's own unique id).
        opp_id = opponent_env_id if opponent_env_id >= 0 else env_id
        env = CustomEnv.create_single_agent_env(
            {"battle_format": battle_format},
            env_id=opp_id,
            use_opponent_cycle=use_opponent_cycle
        )
        env = Monitor(env)
        return env
    return _init


def make_vec_env(n_envs: int = N_ENVS, use_subproc: bool = True, battle_format: str = "gen2randombattle", use_opponent_cycle: bool = True, fixed_env_id: int = -1):
    env_fns = [make_env_fn(env_id=i, seed=42, battle_format=battle_format, use_opponent_cycle=use_opponent_cycle, opponent_env_id=fixed_env_id) for i in range(n_envs)]
    opp_mode = "Heuristics/MaxDamage cycle" if use_opponent_cycle else "Random only"
    if use_subproc and n_envs > 1:
        print(f"[VecEnv] Launching {n_envs} parallel environments (SubprocVecEnv) | format={battle_format} | opponents={opp_mode}")
        return SubprocVecEnv(env_fns, start_method="spawn")
    else:
        print(f"[VecEnv] Using DummyVecEnv with {n_envs} env(s) | format={battle_format} | opponents={opp_mode}")
        return DummyVecEnv(env_fns)