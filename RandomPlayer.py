import asyncio
from poke_env.player import RandomPlayer

async def main():
    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")

    await p1.battle_against(p2, n_battles=1)

    print("P1 wins:", p1.n_won_battles)
    print("P2 wins:", p2.n_won_battles)

asyncio.run(main())
