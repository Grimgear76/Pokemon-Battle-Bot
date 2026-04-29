import asyncio
import time
import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from poke_env.data import GenData
from poke_env.player import Player
from poke_env.environment import SingleAgentWrapper
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from constants import (
    NET_ARCH, LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS,
    ENT_COEF, VF_COEF, CLIP_RANGE, GAE_LAMBDA, N_ENVS, OBS_SIZE,
    ACTION_SPACE_SIZE, MAX_SPEED, STATUS_MAP, SPECIES_NUM_SCALE,
    TARGET_KL, GAMMA, USE_SLOT_EQUIVARIANT, N_SLOTS, SLOT_DIM,
    OWN_SLOT_OFFSET, OPP_SLOT_OFFSET, SLOT_BLOCK_LEN, SLOT_HIDDEN,
    model_path
)
from environment import (
    CustomEnv,
    make_vec_env,
    ProgressCallback,
    EntropyMonitorCallback,
    mask_env,
    _encode_active_mon,
    _encode_boosts,
    _encode_status_flags,
    linear_lr_schedule,
    _is_move_allowed,
    _is_asleep,
    _is_frozen,
    embed_battle_impl,
)

# -----------------------------
# Slot-equivariant feature extractor (Agent19+)
# -----------------------------
class SlotEquivariantExtractor(BaseFeaturesExtractor):
    """
    Slot-equivariant feature extractor for Pokemon obs.

    Splits the 712-dim obs into:
      - own_slot_features [252]  → 6 slots × 42 dims (type1, type2, base_stats)
      - opp_slot_features [252]  → 6 slots × 42 dims
      - rest [208]              → all global / active / scalar features

    Applies a SHARED 2-layer MLP per slot (42 → SLOT_HIDDEN), encoding each
    Pokemon-slot into the same embedding space regardless of which slot it
    sits in. Concatenates the per-slot embeddings with the rest of the obs.

    Output dim: 208 + 2 * N_SLOTS * SLOT_HIDDEN  (= 208 + 384 = 592 with
    defaults). The downstream pi/vf MLPs (NET_ARCH) operate on this output.

    Why: Agent16-18 fed the full 252+252-dim slot blocks into a flat MLP, so
    the policy had to learn a separate 42-dim Pokemon encoding for each of
    the 12 slot positions. That capacity waste shows up as "value head
    excellent, policy stuck" — the value net can lean on the rest of the
    obs (matchup, hp, alive flags), but the policy needs slot-symmetric
    reasoning to decide WHICH bench mon to switch to. Sharing one MLP across
    slots gives the policy exactly that symmetry for free.
    """

    def __init__(self, observation_space, slot_hidden: int = SLOT_HIDDEN):
        n_features = int(np.prod(observation_space.shape))
        assert n_features == OBS_SIZE, (
            f"SlotEquivariantExtractor expects {OBS_SIZE}-dim obs, got {n_features}"
        )
        non_slot_dim = n_features - 2 * SLOT_BLOCK_LEN
        out_dim = non_slot_dim + 2 * N_SLOTS * slot_hidden
        super().__init__(observation_space, features_dim=out_dim)

        self._non_slot_dim = non_slot_dim
        self._slot_hidden  = slot_hidden

        # Shared 2-layer MLP applied to every slot (own + opp).
        self.slot_mlp = nn.Sequential(
            nn.Linear(SLOT_DIM, slot_hidden),
            nn.ReLU(),
            nn.Linear(slot_hidden, slot_hidden),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs: (B, 712)
        own_slots = obs[:, OWN_SLOT_OFFSET : OWN_SLOT_OFFSET + SLOT_BLOCK_LEN]
        opp_slots = obs[:, OPP_SLOT_OFFSET : OPP_SLOT_OFFSET + SLOT_BLOCK_LEN]
        # rest = everything OUTSIDE the two slot blocks (head + tail).
        head = obs[:, :OWN_SLOT_OFFSET]
        tail = obs[:, OPP_SLOT_OFFSET + SLOT_BLOCK_LEN :]

        own_slots = own_slots.view(-1, N_SLOTS, SLOT_DIM)
        opp_slots = opp_slots.view(-1, N_SLOTS, SLOT_DIM)

        own_emb = self.slot_mlp(own_slots).flatten(1)   # (B, N_SLOTS * SLOT_HIDDEN)
        opp_emb = self.slot_mlp(opp_slots).flatten(1)

        return th.cat([head, tail, own_emb, opp_emb], dim=1)


# -----------------------------
# Policy kwargs helper
# -----------------------------
def make_policy_kwargs() -> dict:
    """
    Policy kwargs used by build_model (new training). Loading existing models
    via MaskablePPO.load() does NOT pass this — SB3 reconstructs the policy
    from the saved metadata, which keeps Agent14-18 (no extractor) and
    Agent19+ (with extractor) loadable through the same code path.
    """
    kwargs = dict(net_arch=NET_ARCH)
    if USE_SLOT_EQUIVARIANT:
        kwargs["features_extractor_class"]  = SlotEquivariantExtractor
        kwargs["features_extractor_kwargs"] = dict(slot_hidden=SLOT_HIDDEN)
    return kwargs


# -----------------------------
# Build model
# -----------------------------
def build_model(env, tensorboard_log="./tensorboard_logs/", model_name="model"):
    return MaskablePPO(
        "MlpPolicy",  # ActionMasker in this version of sb3_contrib keeps obs as flat Box — MlpPolicy is correct
        env,
        verbose=0,
        learning_rate=linear_lr_schedule(LEARNING_RATE),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        vf_coef=VF_COEF,
        target_kl=TARGET_KL,
        tensorboard_log=tensorboard_log,
        policy_kwargs=make_policy_kwargs(),
    )


# -----------------------------
# Shared helper: walk wrapper chain to find CustomEnv
# -----------------------------
def _get_custom_env(vec_env, i: int = 0):
    inner = vec_env.envs[i]
    while inner is not None:
        if hasattr(inner, 'eval_wins'):
            return inner
        inner = getattr(inner, 'env', None)
    return None


# -----------------------------
# Train new
# -----------------------------
def train_new(model_name: str, timesteps: int, n_envs: int = N_ENVS, use_subproc: bool = True, battle_format: str = "gen2randombattle"):
    path = model_path(model_name)
    if path.exists():
        print(f"Model '{model_name}' already exists! Delete it or choose another name.")
        return

    print(f"[Training Started] model={model_name}, timesteps={timesteps:,}, n_envs={n_envs}, format={battle_format}")
    # New models train against Random only for broad exploration
    train_env = make_vec_env(n_envs=n_envs, use_subproc=use_subproc, battle_format=battle_format, use_opponent_cycle=False)
    model = build_model(train_env, model_name=model_name)

    checkpoint_dir = model_path(model_name).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = CallbackList([
        ProgressCallback(timesteps, n_envs=n_envs),
        EntropyMonitorCallback(),
        CheckpointCallback(save_freq=25_000, save_path=str(checkpoint_dir), name_prefix=model_name),
    ])

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=model_name
    )
    model.save(path)
    train_env.close()
    print(f"[Model Saved] {model_name}")


# -----------------------------
# Continue training
# -----------------------------
def train_continue(model_name: str, timesteps: int, n_envs: int = N_ENVS, use_subproc: bool = True, battle_format: str = "gen2randombattle"):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found! Train a new model first.")
        return

    print(f"[Continuing Training] model={model_name}, additional timesteps={timesteps:,}, n_envs={n_envs}, format={battle_format}")
    # Continuation uses opponent cycle (Heuristics/MaxDamage) to prevent forgetting
    train_env = make_vec_env(n_envs=n_envs, use_subproc=use_subproc, battle_format=battle_format, use_opponent_cycle=True)
    # No policy_kwargs override — SB3 reconstructs from the saved metadata so
    # Agent14-18 (flat MLP) and Agent19+ (slot-equivariant extractor) both load
    # correctly through this same code path.
    model = MaskablePPO.load(
        path,
        env=train_env,
        tensorboard_log="./tensorboard_logs/",
    )

    model.learning_rate = LEARNING_RATE  # constant for continuation: avoids fractional LR from schedule
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE
    model.ent_coef = ENT_COEF  # override entropy coef from constants.py (saved model may have stale value)
    model.vf_coef = VF_COEF    # override vf_coef so value learning gets the updated weight
    model.gae_lambda = GAE_LAMBDA  # 0.95 → 0.92: tighter lambda reduces advantage variance
    model.n_epochs = N_EPOCHS      # 10 → 12: kl well below target, more updates per rollout
    model.clip_range = lambda _: CLIP_RANGE  # override saved clip_range (sb3 stores it as a schedule callable)
    model.target_kl = TARGET_KL    # override target_kl from constants.py
    model.gamma = GAMMA            # Agent19: 0.99 → 0.995, longer credit assignment

    checkpoint_dir = model_path(model_name).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = CallbackList([
        ProgressCallback(timesteps, n_envs=n_envs),
        EntropyMonitorCallback(),
        CheckpointCallback(save_freq=25_000, save_path=str(checkpoint_dir), name_prefix=model_name),
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


# -----------------------------
# Eval vs built-in opponent
# -----------------------------
def eval_model(model_name: str, n_battles: int = 100, battle_format: str = "gen2randombattle", opponent: str = "max"):
    """
    opponent: "max" | "heuristic" | "random"
    """
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found!")
        return

    # Map the string to a use_opponent_cycle flag and a fixed env_id so only
    # one opponent class is selected: env_id=0 → Heuristics, env_id=1 → MaxDamage.
    opponent = opponent.lower()
    if opponent in ("max", "maxdamage"):
        use_cycle, eval_env_id = True, 1   # env_id 1 % 2 = MaxDamage
    elif opponent in ("heuristic", "heuristics", "simple"):
        use_cycle, eval_env_id = True, 0   # env_id 0 % 2 = Heuristics
    else:  # "random"
        use_cycle, eval_env_id = False, 0

    print(f"[Evaluating] model={model_name}, battles={n_battles}, format={battle_format}, opponent={opponent}")
    eval_env = make_vec_env(n_envs=1, use_subproc=False, battle_format=battle_format, use_opponent_cycle=use_cycle, fixed_env_id=eval_env_id)
    model = MaskablePPO.load(path, env=eval_env)

    wins, losses, draws = 0, 0, 0
    pbar = tqdm(total=n_battles, desc="Evaluating", unit="battles")
    obs = eval_env.reset()
    battles_done = 0

    custom_env = _get_custom_env(eval_env)
    prev_w = custom_env.eval_wins   if custom_env else 0
    prev_l = custom_env.eval_losses if custom_env else 0
    prev_d = custom_env.eval_draws  if custom_env else 0

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
                if custom_env is not None:
                    wins   = custom_env.eval_wins   - prev_w
                    losses = custom_env.eval_losses - prev_l
                    draws  = custom_env.eval_draws  - prev_d

                battles_done += 1
                pbar.update(1)
                pbar.set_postfix(W=wins, L=losses, D=draws, WR=f"{wins/battles_done:.1%}")

    pbar.close()
    eval_env.close()
    print(f"\n[Results] Battles: {n_battles} | Wins: {wins} | Losses: {losses} | Draws: {draws} | Win Rate: {wins/n_battles:.1%}")
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / n_battles}


# -----------------------------
# Eval model1 vs model2
# -----------------------------
def eval_model_vs_model(
    challenger_name: str,
    opponent_name: str,
    n_battles: int = 100,
    battle_format: str = "gen2randombattle"
):
    challenger_path = model_path(challenger_name)
    opponent_path   = model_path(opponent_name)

    if not challenger_path.exists():
        print(f"Challenger model '{challenger_name}' not found!")
        return
    if not opponent_path.exists():
        print(f"Opponent model '{opponent_name}' not found!")
        return

    print(
        f"[Model vs Model] {challenger_name} vs {opponent_name} | "
        f"battles={n_battles} | format={battle_format}"
    )

    frozen_opponent_model = MaskablePPO.load(opponent_path)

    def make_vs_env(env_id: int = 0):
        def _init():
            agent_config    = AccountConfiguration(f"agent_{env_id}", None)
            opponent_config = AccountConfiguration(f"Opponent_bot_{env_id}", None)

            env = CustomEnv(
                battle_format=battle_format,
                log_level=30,
                open_timeout=60,
                strict=False,
                account_configuration1=agent_config,
                account_configuration2=opponent_config,
                server_configuration=LocalhostServerConfiguration,
            )

            opponent = InferenceAgent(
                model=frozen_opponent_model,
                battle_format=battle_format,
                account_configuration=opponent_config,
                server_configuration=LocalhostServerConfiguration,
                start_listening=False,
            )

            base_env = SingleAgentWrapper(env, opponent)
            return Monitor(ActionMasker(base_env, mask_env))
        return _init

    eval_env = DummyVecEnv([make_vs_env(env_id=0)])
    challenger_model = MaskablePPO.load(challenger_path, env=eval_env)

    wins, losses, draws = 0, 0, 0
    pbar = tqdm(total=n_battles, desc=f"{challenger_name} vs {opponent_name}", unit="battles")
    obs = eval_env.reset()
    battles_done = 0

    custom_env = _get_custom_env(eval_env)
    prev_w = custom_env.eval_wins   if custom_env else 0
    prev_l = custom_env.eval_losses if custom_env else 0
    prev_d = custom_env.eval_draws  if custom_env else 0

    while battles_done < n_battles:
        action_masks = get_action_masks(eval_env)
        action, _ = challenger_model.predict(obs, action_masks=action_masks, deterministic=True)
        try:
            obs, reward, dones, infos = eval_env.step(action)
        except AssertionError:
            obs = eval_env.reset()
            continue

        for i, done in enumerate(dones):
            if done:
                if custom_env is not None:
                    wins   = custom_env.eval_wins   - prev_w
                    losses = custom_env.eval_losses - prev_l
                    draws  = custom_env.eval_draws  - prev_d

                battles_done += 1
                pbar.update(1)
                pbar.set_postfix(W=wins, L=losses, D=draws, WR=f"{wins/battles_done:.1%}")

    pbar.close()
    eval_env.close()
    print(
        f"\n[Results] {challenger_name} vs {opponent_name} | "
        f"Battles: {n_battles} | Wins: {wins} | Losses: {losses} | Draws: {draws} | "
        f"Win Rate: {wins/n_battles:.1%}"
    )
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / n_battles}


# -----------------------------
# Inference Agent
# -----------------------------
class InferenceAgent(Player):
    """
    Standalone inference player used for model-vs-model eval and human play.
    _embed() must stay in sync with CustomEnv.embed_battle() in environment.py.
    Current layout: 712 dims (see constants.py OBS_SIZE comment for breakdown).
    """
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._gen_data = GenData.from_gen(2)

    def _embed(self, battle) -> np.ndarray:
        """
        Inference-side observation builder. Delegates to environment.embed_battle_impl —
        the SAME function used by training (CustomEnv.embed_battle). This is the only
        way to guarantee P1 and P2 see byte-identical 712-dim vectors. Do not re-implement.
        """
        return embed_battle_impl(battle, self._gen_data)

    def _build_mask(self, battle) -> np.ndarray:
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if battle.active_pokemon is None:
            return action_mask
        available_moves    = set(battle.available_moves)
        available_switches = set(battle.available_switches)
        moves = list(battle.active_pokemon.moves.values())
        team  = list(battle.team.values())
        if (len(available_moves) == 0 and len(available_switches) == 0) or battle.active_pokemon.must_recharge:
            action_mask[10] = 1
            return action_mask
        own_mon = battle.active_pokemon
        own_incapacitated = _is_asleep(own_mon) or _is_frozen(own_mon)
        boosts = own_mon.boosts if own_mon.boosts is not None else {}
        if not battle.force_switch or not battle.active_pokemon.fainted:
            for slot, move in enumerate(moves):
                if move not in available_moves:
                    continue
                if _is_move_allowed(move, own_mon, battle, boosts, own_incapacitated):
                    action_mask[slot + 6] = 1
        allow_switch = battle.force_switch or own_incapacitated or len(available_moves) > 0
        if allow_switch:
            for slot, mon in enumerate(team):
                if mon in available_switches:
                    action_mask[slot] = 1
        if not any(action_mask):
            action_mask[10] = 1
        return action_mask

    def choose_move(self, battle):
        obs  = self._embed(battle)
        mask = self._build_mask(battle)
        action, _ = self._model.predict(obs[np.newaxis, :], action_masks=mask[np.newaxis, :], deterministic=True)
        action = int(action[0])

        available_moves    = set(battle.available_moves)
        available_switches = set(battle.available_switches)
        moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
        team  = list(battle.team.values())

        if action == 10:
            return self.choose_random_move(battle)
        elif action < 6:
            if action < len(team) and team[action] in available_switches:
                return self.create_order(team[action])
            return self.choose_random_move(battle)
        else:
            move_slot = action - 6
            if move_slot < len(moves) and moves[move_slot] in available_moves:
                return self.create_order(moves[move_slot])
            return self.choose_random_move(battle)


# -----------------------------
# Async Inference Agent (websocket / human play only)
# -----------------------------
class AsyncInferenceAgent(InferenceAgent):
    """Async choose_move for poke-env websocket use (play vs human)."""
    async def choose_move(self, battle):
        return InferenceAgent.choose_move(self, battle)


# -----------------------------
# Play vs Human
# -----------------------------
async def play_vs_human(model_name: str, human_username: str, n_battles: int = 1, battle_format: str = "gen2randombattle"):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found!")
        return

    print(f"[Human Mode] Loading {model_name}...")
    model = MaskablePPO.load(path)

    agent = AsyncInferenceAgent(
        model=model,
        battle_format=battle_format,
        account_configuration=AccountConfiguration("PokemonBot", None),
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
    )

    print(f"[Human Mode] Bot logged in as 'PokemonBot' on localhost:8000")
    print(f"[Human Mode] Open http://localhost:8000 in your browser")
    print(f"[Human Mode] Log in as '{human_username}' and challenge 'PokemonBot'")
    print(f"[Human Mode] Waiting for {n_battles} battle(s) in format '{battle_format}'...")

    await agent.accept_challenges(human_username, n_battles)

    wins   = sum(1 for b in agent.battles.values() if b.won is True)
    losses = sum(1 for b in agent.battles.values() if b.won is False)
    draws  = sum(1 for b in agent.battles.values() if b.won is None)
    print(f"\n[Results] Bot: {wins}W / {losses}L / {draws}D")


# -----------------------------
# League / Ladder Training
# -----------------------------
def train_vs_opponent(learner_name: str, opponent_name: str, timesteps: int, n_envs: int = N_ENVS, use_subproc: bool = True, battle_format: str = "gen2randombattle"):
    learner_path  = model_path(learner_name)
    opponent_path = model_path(opponent_name)

    if not opponent_path.exists():
        print(f"[League] Opponent model '{opponent_name}' not found!")
        return

    print(f"[League] {learner_name} vs {opponent_name} | timesteps={timesteps:,} | n_envs={n_envs} | format={battle_format}")

    frozen_opponent_model = MaskablePPO.load(opponent_path)

    def make_league_env_fn(env_id: int, seed: int = 0):
        def _init():
            set_random_seed(seed + env_id)

            agent_config    = AccountConfiguration(f"agent_{env_id}", None)
            opponent_config = AccountConfiguration(f"Opponent_bot_{env_id}", None)

            env = CustomEnv(
                battle_format=battle_format,
                log_level=30,
                open_timeout=60,
                strict=False,
                account_configuration1=agent_config,
                account_configuration2=opponent_config,
                server_configuration=LocalhostServerConfiguration,
            )

            opponent = InferenceAgent(
                model=frozen_opponent_model,
                battle_format=battle_format,
                account_configuration=opponent_config,
                server_configuration=LocalhostServerConfiguration,
                start_listening=False,
            )

            base_env = SingleAgentWrapper(env, opponent)
            return Monitor(ActionMasker(base_env, mask_env))
        return _init

    env_fns   = [make_league_env_fn(env_id=i, seed=42) for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns, start_method="spawn") if use_subproc and n_envs > 1 else DummyVecEnv(env_fns)

    if learner_path.exists():
        print(f"[League] Resuming {learner_name} from checkpoint")
        model = MaskablePPO.load(
            learner_path,
            env=train_env,
            tensorboard_log="./tensorboard_logs/",
        )
        model.learning_rate = LEARNING_RATE  # constant for continuation: avoids fractional LR from schedule
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE
        model.ent_coef = ENT_COEF  # override entropy coef from constants.py (saved model may have stale value)
        model.vf_coef = VF_COEF    # override vf_coef so value learning gets the updated weight
        model.gae_lambda = GAE_LAMBDA
        model.n_epochs = N_EPOCHS
        model.gamma = GAMMA
        reset_timesteps = False
    else:
        print(f"[League] Creating new model {learner_name}")
        model = build_model(train_env, model_name=learner_name)
        reset_timesteps = True

    callbacks = CallbackList([
        ProgressCallback(timesteps, n_envs=n_envs),
        EntropyMonitorCallback(),
    ])

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
        tb_log_name=learner_name
    )
    model.save(learner_path)
    train_env.close()
    print(f"[League] Saved {learner_name}")


# -----------------------------
# Self-Play training
# -----------------------------
# Builtin opponent tokens accepted in the opponent_pool list (case-insensitive).
_BUILTIN_OPPONENTS = {"random", "heuristic", "heuristics", "max", "maxdamage"}


def _make_self_play_env_fn(
    env_id: int,
    opponent_token: str,
    battle_format: str,
    seed: int = 42,
):
    """
    Build an env factory whose opponent is decided by `opponent_token`:
      - "random" / "heuristic" / "max"  → builtin scripted player
      - any other string                → InferenceAgent loaded from models/<token>.zip
    The model is loaded INSIDE the subprocess so we don't pickle it across the
    SubprocVecEnv boundary (which would force every model in the pool through
    the parent process even if only one env uses it).
    """
    token_lower = opponent_token.lower()
    is_builtin = token_lower in _BUILTIN_OPPONENTS
    opp_path = None if is_builtin else str(model_path(opponent_token))

    def _init():
        # Imports happen in the subprocess so _RUN_SUFFIX is the SUBPROCESS's
        # freshly-generated suffix (per-process unique usernames), not the
        # parent's. environment.py is re-imported when SubprocVecEnv spawns.
        from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
        from environment import MaxDamagePlayer, _RUN_SUFFIX

        set_random_seed(seed + env_id)
        # Stagger startup — same reason as make_env_fn in environment.py.
        time.sleep(env_id * 2.0)

        agent_config    = AccountConfiguration(f"agt_{env_id}_{_RUN_SUFFIX}", None)
        opponent_config = AccountConfiguration(f"opp_{env_id}_{_RUN_SUFFIX}", None)

        env = CustomEnv(
            battle_format=battle_format,
            log_level=30,
            open_timeout=60,
            challenge_timeout=180.0,
            strict=False,
            account_configuration1=agent_config,
            account_configuration2=opponent_config,
            server_configuration=LocalhostServerConfiguration,
        )

        if is_builtin:
            if token_lower == "random":
                opponent = RandomPlayer(start_listening=False, account_configuration=opponent_config)
            elif token_lower in ("heuristic", "heuristics"):
                opponent = SimpleHeuristicsPlayer(start_listening=False, account_configuration=opponent_config)
            else:  # max / maxdamage
                opponent = MaxDamagePlayer(start_listening=False, account_configuration=opponent_config)
        else:
            # Frozen learner — load with NO policy_kwargs override so any saved arch
            # (Agent14-18 flat MLP, Agent19+ slot-equivariant) reconstructs cleanly.
            opp_model = MaskablePPO.load(opp_path)
            opponent = InferenceAgent(
                model=opp_model,
                battle_format=battle_format,
                account_configuration=opponent_config,
                server_configuration=LocalhostServerConfiguration,
                start_listening=False,
            )

        base_env = SingleAgentWrapper(env, opponent)
        return Monitor(ActionMasker(base_env, mask_env))

    return _init


def train_self_play(
    learner_name: str,
    timesteps: int,
    opponent_pool: list,
    n_envs: int = N_ENVS,
    use_subproc: bool = True,
    battle_format: str = "gen2randombattle",
):
    """
    Train `learner_name` against a pool of opponents distributed round-robin
    across n_envs. Each pool entry is one of:
      - "random" / "heuristic" / "max"  → builtin scripted player
      - <model_name>                    → InferenceAgent loaded from models/<name>.zip

    Mixing builtins with frozen past learners (e.g.
    ["Agent14", "Agent17", "Agent18", "heuristic", "max"]) is recommended:
    the builtins anchor the policy against forgetting, while the frozen
    learners push it past the ~55%/45% Heuristics/MaxDamage ceiling that
    Agent14-18 all bumped into.
    """
    learner_path = model_path(learner_name)

    # Validate pool: drop missing model checkpoints up-front.
    valid_pool: list = []
    for entry in opponent_pool:
        if entry.lower() in _BUILTIN_OPPONENTS:
            valid_pool.append(entry)
            continue
        if model_path(entry).exists():
            valid_pool.append(entry)
        else:
            print(f"[SelfPlay] Skipping missing opponent: {entry}")
    if not valid_pool:
        print("[SelfPlay] No valid opponents in pool!")
        return

    print(
        f"[SelfPlay] {learner_name} vs pool {valid_pool} | timesteps={timesteps:,} | "
        f"n_envs={n_envs} | format={battle_format}"
    )

    env_fns = [
        _make_self_play_env_fn(
            env_id=i,
            opponent_token=valid_pool[i % len(valid_pool)],
            battle_format=battle_format,
            seed=42,
        )
        for i in range(n_envs)
    ]
    train_env = SubprocVecEnv(env_fns, start_method="spawn") if use_subproc and n_envs > 1 else DummyVecEnv(env_fns)

    if learner_path.exists():
        print(f"[SelfPlay] Resuming {learner_name} from checkpoint")
        model = MaskablePPO.load(
            learner_path,
            env=train_env,
            tensorboard_log="./tensorboard_logs/",
        )
        model.learning_rate = LEARNING_RATE
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE
        model.ent_coef   = ENT_COEF
        model.vf_coef    = VF_COEF
        model.gae_lambda = GAE_LAMBDA
        model.n_epochs   = N_EPOCHS
        model.clip_range = lambda _: CLIP_RANGE
        model.target_kl  = TARGET_KL
        model.gamma      = GAMMA
        reset_timesteps  = False
    else:
        print(f"[SelfPlay] Creating new model {learner_name}")
        model = build_model(train_env, model_name=learner_name)
        reset_timesteps = True

    checkpoint_dir = learner_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = CallbackList([
        ProgressCallback(timesteps, n_envs=n_envs),
        EntropyMonitorCallback(),
        CheckpointCallback(save_freq=25_000, save_path=str(checkpoint_dir), name_prefix=learner_name),
    ])

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
        tb_log_name=learner_name,
    )
    model.save(learner_path)
    train_env.close()
    print(f"[SelfPlay] Saved {learner_name}")