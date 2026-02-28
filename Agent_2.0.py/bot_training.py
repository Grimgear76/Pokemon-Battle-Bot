import asyncio
import numpy as np
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from poke_env.data import GenData
from poke_env.player import Player
from poke_env.environment import SingleAgentWrapper
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from constants import (
    NET_ARCH, LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS,
    ENT_COEF, CLIP_RANGE, GAE_LAMBDA, N_ENVS, OBS_SIZE,
    ACTION_SPACE_SIZE, MAX_SPEED, STATUS_MAP, model_path
)
from environment import (
    make_vec_env, ProgressCallback, EntropyMonitorCallback,
    CustomEnv, mask_env, _is_frozen, _is_paralyzed
)


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
        vf_coef=0.3, 
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=NET_ARCH)
    )


# -----------------------------
# Train new
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


# -----------------------------
# Continue training
# -----------------------------
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


# -----------------------------
# Eval
# -----------------------------
def eval_model(model_name: str, n_battles: int = 100):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found!")
        return

    print(f"[Evaluating] model={model_name}, battles={n_battles}")
    eval_env = make_vec_env(n_envs=1, use_subproc=False)
    model = MaskablePPO.load(path, env=eval_env, policy_kwargs=dict(net_arch=NET_ARCH))

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
# Inference Agent (sync — works inside SB3 env loop)
# -----------------------------
class InferenceAgent(Player):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._gen_data = GenData.from_gen(1)
        self._species_to_id = {name: i for i, name in enumerate(self._gen_data.pokedex.keys())}
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
        features[0] = -1.0 if (mon.fainted or mon.max_hp == 0) else (mon.current_hp / mon.max_hp) * 2 - 1
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
            moves_n, pokemon_team = 4, 6
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
                team_hp_ratio[i] = -1.0 if (mon.fainted or mon.max_hp == 0) else (mon.current_hp / mon.max_hp) * 2 - 1
            for i, (_, mon) in enumerate(sorted(battle.opponent_team.items())):
                opponent_hp_ratio[i] = -1.0 if (mon.fainted or mon.max_hp == 0) else (mon.current_hp / mon.max_hp) * 2 - 1
            for i, move in enumerate(battle.available_moves):
                try:
                    moves_base_power[i] = (move.base_power / 250) * 2 - 1 if move.base_power else 0.0
                    if battle.opponent_active_pokemon:
                        moves_dmg_multiplier[i] = np.clip(
                            move.type.damage_multiplier(
                                battle.opponent_active_pokemon.type_1,
                                battle.opponent_active_pokemon.type_2,
                                type_chart=self._gen_data.type_chart
                            ) - 1.0, -1.0, 1.0)
                    moves_pp_ratio[i] = (move.current_pp / move.max_pp) * 2 - 1
                except AssertionError:
                    pass
            for i, mon in enumerate(battle.team.values()):
                if mon.fainted: self_team_status[i] = -1.0
            for i, mon in enumerate(battle.opponent_team.values()):
                if mon.fainted: opponent_team_status[i] = -1.0
            for i, (_, mon) in enumerate(sorted(battle.team.items())):
                if mon and mon.species:
                    idx = self._species_to_id.get(mon.species.lower(), 0)
                    team_identifier[i] = (idx / (len(self._species_to_id) - 1)) * 2 - 1
            for i, (_, mon) in enumerate(sorted(battle.opponent_team.items())):
                if mon and mon.species:
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
            opp_move_eff = np.zeros(4, dtype=np.float32)
            if battle.opponent_active_pokemon and battle.active_pokemon:
                for i, move in enumerate(list(battle.opponent_active_pokemon.moves.values())[:4]):
                    try:
                        eff = move.type.damage_multiplier(
                            battle.active_pokemon.type_1, battle.active_pokemon.type_2,
                            type_chart=self._gen_data.type_chart)
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
        if (len(available_moves) == 0 and len(available_switches) == 0) or battle.active_pokemon.must_recharge:
            action_mask[10] = 1
            return action_mask
        if not battle.force_switch or not battle.active_pokemon.fainted:
            for slot, move in enumerate(moves):
                if move in available_moves:
                    action_mask[slot + 6] = 1
        for slot, mon in enumerate(team):
            if mon in available_switches:
                action_mask[slot] = 1
        if not any(action_mask):
            action_mask[10] = 1
        return action_mask

    def choose_move(self, battle):
        """Sync — used by SingleAgentWrapper inside SB3 env loop."""
        obs = self._embed(battle)
        mask = self._build_mask(battle)
        action, _ = self._model.predict(obs[np.newaxis, :], action_masks=mask[np.newaxis, :], deterministic=True)
        action = int(action[0])

        available_moves = set(battle.available_moves)
        available_switches = set(battle.available_switches)
        moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
        team = list(battle.team.values())

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
async def play_vs_human(model_name: str, human_username: str, n_battles: int = 1):
    path = model_path(model_name)
    if not path.exists():
        print(f"Model '{model_name}' not found!")
        return

    print(f"[Human Mode] Loading {model_name}...")
    model = MaskablePPO.load(path, policy_kwargs=dict(net_arch=NET_ARCH))

    agent = AsyncInferenceAgent(
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

    wins   = sum(1 for b in agent.battles.values() if b.won is True)
    losses = sum(1 for b in agent.battles.values() if b.won is False)
    draws  = sum(1 for b in agent.battles.values() if b.won is None)
    print(f"\n[Results] Bot: {wins}W / {losses}L / {draws}D")


# -----------------------------
# League / Ladder Training
# -----------------------------
# Train LEARNER_NAME against a frozen OPPONENT_NAME.
# The learner trains via SB3 (parallel envs supported).
# The opponent is frozen — it only runs inference, never updates.
#
# Ladder usage:
#   train_vs_opponent("Gen1", "ParallelTest8", 500_000)  # Gen1 starts from Test8 weights, trains vs frozen Test8
#   train_vs_opponent("Gen2", "Gen1",          500_000)  # Gen2 learns vs frozen Gen1
#   train_vs_opponent("Gen3", "Gen2",          500_000)  # and so on...
# -----------------------------
def train_vs_opponent(learner_name: str, opponent_name: str, timesteps: int, n_envs: int = N_ENVS, use_subproc: bool = True):
    learner_path  = model_path(learner_name)
    opponent_path = model_path(opponent_name)

    if not opponent_path.exists():
        print(f"[League] Opponent model '{opponent_name}' not found!")
        return

    print(f"[League] {learner_name} vs {opponent_name} | timesteps={timesteps:,} | n_envs={n_envs}")

    # Load frozen opponent once — shared across all envs (inference only, no gradients)
    frozen_opponent_model = MaskablePPO.load(opponent_path, policy_kwargs=dict(net_arch=NET_ARCH))

    def make_league_env_fn(env_id: int, seed: int = 0):
        def _init():
            set_random_seed(seed + env_id)

            agent_config    = AccountConfiguration(f"agent_{env_id}", None)
            opponent_config = AccountConfiguration(f"Opponent_bot_{env_id}", None)

            env = CustomEnv(
                battle_format="gen1randombattle",
                log_level=30,
                open_timeout=None,
                strict=False,
                account_configuration1=agent_config,
                account_configuration2=opponent_config,
                server_configuration=LocalhostServerConfiguration,
            )

            # Sync InferenceAgent — SingleAgentWrapper requires sync choose_move
            opponent = InferenceAgent(
                model=frozen_opponent_model,
                battle_format="gen1randombattle",
                account_configuration=opponent_config,
                server_configuration=LocalhostServerConfiguration,
                start_listening=False,
            )

            base_env = SingleAgentWrapper(env, opponent)
            return Monitor(ActionMasker(base_env, mask_env))
        return _init

    env_fns   = [make_league_env_fn(env_id=i, seed=42) for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns, start_method="spawn") if use_subproc and n_envs > 1 else DummyVecEnv(env_fns)

    # Load existing learner checkpoint or create fresh
    if learner_path.exists():
        print(f"[League] Resuming {learner_name} from checkpoint")
        model = MaskablePPO.load(
            learner_path,
            env=train_env,
            tensorboard_log="./tensorboard_logs/",
            policy_kwargs=dict(net_arch=NET_ARCH)
        )
        model.learning_rate = LEARNING_RATE
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE
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