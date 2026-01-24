# RL

My personal implementations of reinforcement learning algorithms.

---

## Algorithms

### PPO (Proximal Policy Optimization)
> See [ppo/ppo/algorithm.py](ppo/ppo/algorithm.py) for the PPO implementation and [ppo/ppo/gae.py](ppo/ppo/gae.py) for Generalized Advantage Estimation.

### SAC (Soft Actor-Critic)
> See [sac/sac.py](sac/sac.py) for the SAC implementation with discrete action spaces.

---

<p align="center">
    <image src=ppo/snake.gif>
</p>

## Results on `Wandb`!

### PPO
- [Snake](https://wandb.ai/alantian2018/ppo-snake/runs/j1jym4xk)
- [CartPole](https://wandb.ai/alantian2018/ppo-cartpole?nw=nwuseralantian2018)
- [LunarLander](https://wandb.ai/alantian2018/ppo-lunarlander?nw=nwuseralantian2018)
- [Acrobot](https://wandb.ai/alantian2018/ppo-acrobot?nw=nwuseralantian2018)

### SAC
- [Pendulum](https://wandb.ai/alantian2018/sac-cartpole?nw=nwuseralantian2018)

---
Some of the boilerplate (`eg configs, wandb logging, utils, etc`) were handled by opus 4.5 :)
