from dataclasses import dataclass, asdict
from typing import Optional, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.networks import Policy, Qfunction
from sac.replaybuffer import ReplayBuffer, Step
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import wandb


@dataclass
class SACConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    update_every: int = 1
    num_updates: int = 10
    replay_buffer_capacity: int = 1_000_000
    before_training_steps: int = 1000
    gradient_step_ratio: int = 1 # num of gradient steps per rollout step
    collect_rollout_steps: int = 1000
    device: str = "cpu"
    action_low: float = 9999
    action_high: float = 9999
    is_continuous: bool = False
    # Wandb
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    video_log_freq: Optional[int] = None
    log_freq: int = 10  # Log every N training steps


class SACLogger:
    """Handles wandb logging for SAC."""
    
    def __init__(
        self,
        config: SACConfig,
        make_env: Optional[Callable[..., gym.Env]] = None,
    ):
        self.config = config
        self.make_env = make_env
        self.use_wandb = getattr(config, 'wandb_project', None) is not None
        self.video_log_freq = getattr(config, 'video_log_freq', None)
        self.last_video_step = -1
        
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
    
    def log(self, metrics: dict, step: int):
        """Log metrics to wandb."""
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_rollout(self, episode_returns: list, step: int):
        """Log rollout episode statistics."""
        if self.use_wandb and episode_returns:
            self.log({
                "rollout/episode_return_mean": sum(episode_returns) / len(episode_returns),
                "rollout/episode_return_max": max(episode_returns),
                "rollout/num_episodes": len(episode_returns),
            }, step=step)
    
    def log_training(self, qf1_loss: float, qf2_loss: float, policy_loss: float, step: int):
        """Log training metrics."""
        if self.use_wandb:
            self.log({
                "train/qf1_loss": qf1_loss,
                "train/qf2_loss": qf2_loss,
                "train/policy_loss": policy_loss,
            }, step=step)
    
    def maybe_record_video(self, policy: torch.nn.Module, step: int, device: str):
        """Record video if it's time to do so."""
        if not self.use_wandb or self.video_log_freq is None or self.make_env is None:
            return
        
        if step == 0 or step - self.last_video_step >= self.video_log_freq:
            self._record_video(policy, step, device)
            self.last_video_step = step
    
    def _record_video(self, policy: torch.nn.Module, step: int, device: str, num_evals: int = 10):
        eval_env = self.make_env(render_mode="rgb_array")
        
        all_returns = []
        best_return = float('-inf')
        best_frames = None
        
        for _ in range(num_evals):
            frames = []
            obs, _ = eval_env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            
            done = False
            episode_return = 0.0
            while not done:
                frame = eval_env.render()
                frames.append(frame)
                
                with torch.no_grad():
                    action, _ = policy.get_action(obs)
                if not self.config.is_continuous:
                    action =  torch.floor(action).to(torch.int32)
                
                obs, reward, terminated, truncated, _ = eval_env.step(action.cpu().numpy())
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                episode_return += reward
                done = terminated or truncated
            
            all_returns.append(episode_return)
            if episode_return > best_return:
                best_return = episode_return
                best_frames = frames
        
        eval_env.close()
        
        video = np.stack(best_frames).transpose(0, 3, 1, 2)
        self.log({
            "eval/video": wandb.Video(video, fps=30, format="mp4"),
            "eval/episode_return_mean": sum(all_returns) / len(all_returns),
            "eval/episode_return_max": max(all_returns),
            "eval/episode_return_min": min(all_returns),
        }, step=step)
    
    def finish(self):
        """Finish wandb run."""
        if self.use_wandb:
            wandb.finish()


class SAC:
    def __init__(
        self, 
        config: SACConfig, 
        env: gym.Env,
        make_env: Optional[Callable[..., gym.Env]] = None,
    ):
        self.config = config
        self.device = config.device
        
        self.policy = Policy(config.state_dim, config.action_dim, config.action_low, config.action_high, config.hidden_dim).to(self.device)

        self.qf1 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.qf2 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)

        self.target_qf1 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_qf2 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())


        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.optimizer_qf1 = optim.Adam(self.qf1.parameters(), lr=config.critic_lr)
        self.optimizer_qf2 = optim.Adam(self.qf2.parameters(), lr=config.critic_lr)

        self.env = env
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

        self.old_obs, _ = self.env.reset()
        
        # Episode tracking
        self.episode_return = 0.0
        self.episode_returns = []
        
        # Logger
        self.logger = SACLogger(config, make_env=make_env)
        
        
    def collect_rollout(self):
       
        for _ in range(self.config.collect_rollout_steps):
            obs_tensor = torch.tensor(self.old_obs, dtype=torch.float32).to(self.device)
            action, _ = self.policy.get_action(obs_tensor)
            if not self.config.is_continuous:
                action = torch.floor(action).to(torch.int32)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())                
            step = Step(self.old_obs.copy(), 
                        action.detach().cpu().numpy(),
                        reward,
                        next_obs.copy(),
                        terminated,
                        truncated
                    )
            
            self.replay_buffer.add(step)
            self.old_obs = next_obs
            
            # Track episode return
            self.episode_return += reward

            if terminated or truncated:
                self.episode_returns.append(self.episode_return)
                self.episode_return = 0.0
                self.old_obs, _ = self.env.reset()

    def calculate_q_target(self, next_obs: np.ndarray, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray):
        with torch.no_grad():
            dones = terminated | truncated
            
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device).unsqueeze(-1)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(-1)

            sampled_actions, log_probs = self.policy.get_action(next_obs_tensor)
        
            q1 = self.target_qf1(next_obs_tensor, sampled_actions) 
            q2 = self.target_qf2(next_obs_tensor, sampled_actions) 

            soft_v = torch.min(q1, q2) - self.config.alpha * log_probs
        
            return reward_tensor + self.config.gamma * (1 - dones_tensor) * soft_v
    
    def calculate_policy_target(self, obs: np.ndarray):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        actions, log_probs = self.policy.get_action(obs_tensor)
      
        q1 = self.qf1(obs_tensor, actions)
        q2 = self.qf2(obs_tensor, actions)
       
        return torch.min(q1, q2) - (self.config.alpha * log_probs)

    def update_qf(self, qf1_loss, qf2_loss):
        self.optimizer_qf1.zero_grad()
        self.optimizer_qf2.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self.optimizer_qf1.step()
        self.optimizer_qf2.step()

    def update_targets(self):
        tau = self.config.tau
        with torch.no_grad():
            for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

   
    
    def train(self, total_train_steps: int):
        assert self.config.action_low < self.config.action_high, "action_low must be less than action_high"

        if not self.config.is_continuous:
            assert self.config.action_low == 0 and self.config.action_dim == 1, "action_low must be 0 and action dim must be 1"
            self.config.action_high -= 1e-6
            print('using torch.floor for action')
        
        while len(self.replay_buffer) < self.config.before_training_steps:
            self.collect_rollout()
        
        for step in tqdm(range(total_train_steps), desc='Training'):
            # Maybe record video
            self.logger.maybe_record_video(self.policy, step, self.device)
            
            # always collect rollout
            self.episode_returns = []
            self.collect_rollout()
            
            # Log rollout stats
            self.logger.log_rollout(self.episode_returns, step=step)

            # if replay buffer is not full, collect more rollouts
            while len(self.replay_buffer) < self.config.batch_size:
                self.collect_rollout()

            qf1_loss_total = 0.0
            qf2_loss_total = 0.0
            policy_loss_total = 0.0
            
            for _ in range(self.config.gradient_step_ratio):
                obs, action, reward, next_obs, terminated, truncated, _ = self.replay_buffer.sample(self.config.batch_size)
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
                
                q_target = self.calculate_q_target(next_obs, reward, terminated, truncated).detach()

                qf1_loss = F.mse_loss(self.qf1(obs_tensor, action_tensor), q_target)
                qf2_loss = F.mse_loss(self.qf2(obs_tensor, action_tensor), q_target)
                self.update_qf(qf1_loss, qf2_loss)

                policy_target = self.calculate_policy_target(obs)
                # we need to push towards the right way ie minimize the loss here. no mse since we need to move
                # towards this direction, we don't want penalty for being too low.
                policy_loss = -policy_target.mean()
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                self.update_targets()
                
                qf1_loss_total += qf1_loss.item()
                qf2_loss_total += qf2_loss.item()
                policy_loss_total += policy_loss.item()
            
            # Log training metrics
            if step % self.config.log_freq == 0:
                self.logger.log_training(
                    qf1_loss=qf1_loss_total / self.config.gradient_step_ratio,
                    qf2_loss=qf2_loss_total / self.config.gradient_step_ratio,
                    policy_loss=policy_loss_total / self.config.gradient_step_ratio,
                    step=step,
                )
        
        self.logger.finish()
