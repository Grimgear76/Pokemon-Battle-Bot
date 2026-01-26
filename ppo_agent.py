import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from poke_env.player import Player, RandomPlayer

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
    def __init__(self, battle_format="gen9randombattle", log_level=30):
        super().__init__(battle_format=battle_format, log_level=log_level)
        self.policy = PolicyNetwork(input_size=10, hidden_size=64, output_size=4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.last_log_prob = None

    def choose_move(self, battle):
        x = self.battle_to_tensor(battle)
        logits = self.policy(x)
        m = Categorical(logits=logits)
        action_idx = m.sample()
        self.last_log_prob = m.log_prob(action_idx)

        if battle.available_moves:
            move = battle.available_moves[action_idx.item() % len(battle.available_moves)]
            return self.create_order(move)
        return self.choose_random_move(battle)

    def battle_to_tensor(self, battle):
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0
        status = 0
        tensor = torch.tensor([my_hp, opp_hp, status] + [0]*7, dtype=torch.float32)
        return tensor

    async def on_battle_end(self, battle, won):
        """Update policy after battle ends."""
        my_hp_frac = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        opp_hp_frac = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0

        # Base reward
        reward = 1.0 if won else -1.0

        # Add HP difference
        reward += 0.5 * (my_hp_frac - opp_hp_frac)

        # Update policy
        if self.last_log_prob is not None:
            loss = -self.last_log_prob * reward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ---- Training function ----
async def train():
    agent = PPOAgent()
    opponent = RandomPlayer(battle_format="gen9randombattle")
    n_battles = 5  # small for testing

    for i in range(n_battles):
        print(f"Starting battle {i+1}...")
        await agent.battle_against(opponent, n_battles=1)
    print("Training finished.")

# ---- Run ----
if __name__ == "__main__":
    asyncio.run(train())
