import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer


# ---- Neural Network ----
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)


# ---- PPO Agent ----
class PPOAgent(Player):
    def __init__(self, battle_format="gen1randombattle", log_level=30):
        super().__init__(battle_format=battle_format, log_level=log_level)
        self.policy = PolicyNetwork(input_size=10, hidden_size=64, output_size=4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.last_log_prob = None
        self._last_moves = []  # Track moves per battle

    def choose_move(self, battle):
        try:
            x = self.battle_to_tensor(battle)
            logits = self.policy(x)
            m = Categorical(logits=logits)
            action_idx = m.sample()
            self.last_log_prob = m.log_prob(action_idx)

            # Avoid problematic moves
            available = [
                m for m in battle.available_moves
                if m.id not in ("mirrormove", "metronome", "assist", "transform", "copycat")
            ]
            if not available:
                return self.choose_random_move(battle)

            move = available[action_idx.item() % len(available)]
            # Track move info for reward
            self._last_moves.append({
                "move_obj": move,
                "damage_done": getattr(move, "damage_done", 0),
                "damage_taken": getattr(move, "damage_taken", 0),
            })
            return self.create_order(move)

        except Exception as e:
            print("Skipping unhandled move:", e)
            return self.choose_random_move(battle)

    def battle_to_tensor(self, battle):
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0
        status = 0
        # Pad to 10 features (placeholder)
        return torch.tensor([my_hp, opp_hp, status] + [0]*7, dtype=torch.float32)

    def move_effectiveness(self, move, opponent):
        """Return type effectiveness multiplier"""
        try:
            return move.type.damage_multiplier(opponent.types)
        except Exception:
            return 1.0

    async def on_battle_end(self, battle, won):
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0

        # Base reward
        reward = 1.0 if won else -1.0
        reward += 0.5 * (my_hp - opp_hp)

        # Bonus for moves
        for move_info in self._last_moves:
            move = move_info.get("move_obj")
            if move:
                multiplier = self.move_effectiveness(move, battle.opponent_active_pokemon)
                if multiplier > 1.0:
                    reward += 0.03  # super effective bonus
            reward += 0.02 * move_info.get("damage_done", 0)
            reward -= 0.01 * move_info.get("damage_taken", 0)

        # Bonus for fainted opponent Pokémon
        fainted_own = sum(1 for p in battle.team.values() if p.current_hp == 0)
        reward -= 0.05 * fainted_own  # penalize losing Pokémon

        # Update policy
        if self.last_log_prob is not None:
            loss = -self.last_log_prob * reward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear move history
        self._last_moves = []

        # Print summary
        print(f"Battle finished | Won: {won} | Reward: {reward:.2f} | My HP: {my_hp:.2f} | Opponent HP: {opp_hp:.2f} | Fainted: {fainted_enemies}")


# ---- Training function ----
async def train():
    agent = PPOAgent()
    opponent = SimpleHeuristicsPlayer(battle_format="gen1randombattle")

    n_battles = 800

    print("Starting training...\n")
    for i in range(n_battles):
        print(f"Starting battle {i+1}...")
        await agent.battle_against(opponent, n_battles=1)

    print("\nTraining finished!")


# ---- Run ----
if __name__ == "__main__":
    asyncio.run(train())
