from poke_env.player import RandomPlayer
import asyncio
import logging

# Silence all poke-env logs
logging.getLogger("poke_env").setLevel(logging.CRITICAL)

async def main():
    # Create two random players
    p1 = RandomPlayer(battle_format="gen9randombattle", log_level=logging.CRITICAL)
    p2 = RandomPlayer(battle_format="gen9randombattle", log_level=logging.CRITICAL)

    print("Battle starting...", flush=True)

    # Run a single battle
    await p1.battle_against(p2, n_battles=1)

    print("Battle finished", flush=True)
    print(f"P1 Wins: {p1.n_won_battles}")
    print(f"P2 Wins: {p2.n_won_battles}")

if __name__ == "__main__":
    asyncio.run(main())
