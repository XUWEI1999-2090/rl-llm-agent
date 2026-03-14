import asyncio
from src.agents.notebook_agent import make_bixbench_agent, run_episode
from src.envs.notebook_env import BixBenchDataset

async def main():
    # 1. 创建数据集（需要你事先下载好 BixBench 到本地某个目录）
    dataset = BixBenchDataset(
        data_dir="path/to/BixBench/data",  # 替换成你本地 BixBench capsule 路径
        use_docker=False,                  # 本地 notebook 内核测试时可以先关 docker
        split="train",
    )

    # 2. 拿一个 capsule 对应的环境
    env = dataset.get_env(0)

    # 3. 创建 Notebook Agent（用 gpt-4o，不训练）
    agent = make_bixbench_agent(
        model_name="gpt-4o",
        model_base_url=None,   # 如果你有本地 vLLM，可以填 http://localhost:8000/v1
        temperature=0.3,
        max_steps=20,
    )

    # 4. 跑一条 episode
    traj = await run_episode(agent, env, max_steps=20)
    print(traj["total_reward"], traj["n_steps"], traj["done"])

if __name__ == "__main__":
    asyncio.run(main())