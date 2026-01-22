import asyncio
from env.battle_env import BattleEnv
from agent.random_agent import RandomAgent

async def main():
    env = BattleEnv(RandomAgent, RandomAgent, battle_format="gen9randombattle")
    print("Starting 1 battle...")
    await env.run(n_battles=1)
    print("Finished!")
    print(env.get_results())

if __name__ == "__main__":
    asyncio.run(main())
