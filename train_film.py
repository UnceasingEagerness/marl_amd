"""
train_film.py — Training script for Variant 5: STAE + FiLM Goal Encoder + Gated Fusion.
Updated for Multi-GPU (8x MI300X) Data Parallelism using jax.pmap.
"""

import os
import time
import functools
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
from flax.training.train_state import TrainState
from flax import struct
from typing import Any
from orbax.checkpoint import PyTreeCheckpointer
import flax.jax_utils as jax_utils

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import print as rprint

from env.jax_usv_env import JaxUSVEnv, EnvParams, EnvState
from algorithms.flax_sac import Actor_FiLM, SoftQNetwork_FiLM
from algorithms.jax_buffer import JaxReplayBuffer, ReplayBufferState
from algorithms.sac_update import update_critic, update_actor, update_alpha, Transition

console = Console()


@struct.dataclass
class RunnerState:
    env_state:            EnvState
    obs:                  jnp.ndarray
    episode_return:       jnp.ndarray
    actor_state:          TrainState
    critic_state:         TrainState
    target_critic_params: Any
    log_alpha:            jnp.ndarray
    alpha_opt_state:      optax.OptState
    buffer_state:         ReplayBufferState
    rng:                  jax.random.PRNGKey
    step_count:           int


def main():
    num_devices = jax.device_count()
    
    # ── Hyperparameters ───────────────────────────────────────────────────────
    num_envs       = 256  # Increased to feed 8 GPUs
    envs_per_device = num_envs // num_devices
    num_agents     = 5
    total_timesteps = 400_000
    learning_starts = 10_000
    batch_size     = 256
    per_device_batch_size = batch_size // num_devices
    buffer_size    = 100_000
    gamma          = 0.99
    tau_target     = 0.005
    policy_lr      = 3e-4
    q_lr           = 3e-4
    target_entropy = -2.0

    seq_len      = 10
    base_obs_dim = 72 + (num_agents - 1) * 5
    obs_dim      = base_obs_dim * seq_len
    action_dim   = 2
    total_insertions_per_step = envs_per_device * num_agents

    layout = {
        "ego":              {"start": 0,  "dim": 8},
        "goal":             {"start": 0,  "dim": 8},
        "lidar":            {"start": 8,  "dim": 64},
        "auv_entities":     {"start": 72, "dim": (num_agents - 1) * 5,
                             "count": num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5},
    }

    env        = JaxUSVEnv()
    env_params = env.default_params.replace(num_agents=num_agents)

    console.print(Panel.fit(
        f"[bold cyan]RLSim V5 Multi-GPU — FiLM Goal Encoder + Gated Fusion[/bold cyan]\n"
        f"Detected Devices: {num_devices} | Envs/Device: {envs_per_device}",
        border_style="cyan"
    ))
    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Parameter", width=25)
    tbl.add_column("Value", justify="right", style="green")
    tbl.add_row("Num Devices",       str(num_devices))
    tbl.add_row("Num Envs (Total)",  str(num_envs))
    tbl.add_row("Total Timesteps",   f"{total_timesteps:,}")
    tbl.add_row("Batch Size (Total)",str(batch_size))
    tbl.add_row("Batch/Device",      str(per_device_batch_size))
    console.print(tbl)

    rng = jax.random.PRNGKey(42)

    console.print("[bold yellow]Pre-computing Map Bank (1000 Maps)...[/bold yellow]")
    rng, map_key = jax.random.split(rng)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank, currents_bank = jitted_map_gen(
        map_key, int(env_params.num_agents), int(env_params.num_obstacles),
        float(env_params.map_size), int(env_params.map_bank_size),
    )
    jax.block_until_ready(goals_bank)
    env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank, currents_bank=currents_bank)
    console.print("[bold green]✔ Map Bank loaded.[/bold green]")

    # ── Network + Buffer ──────────────────────────────────────────────────────
    actor  = Actor_FiLM(
        layout=layout, action_dim=action_dim,
        action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim),
    )
    critic = SoftQNetwork_FiLM(layout=layout)
    buffer = JaxReplayBuffer(buffer_size, obs_dim, action_dim)

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step  = jax.vmap(env.step,  in_axes=(0, 0, 0, None))

    rng, _rng = jax.random.split(rng)
    reset_keys = jax.random.split(_rng, num_envs)
    init_obs, init_env_state = vmap_reset(reset_keys, env_params)

    # Shard env state and obs to [num_devices, envs_per_device, ...]
    def shard(x):
        return x.reshape((num_devices, envs_per_device) + x.shape[1:])
    
    init_obs = shard(init_obs)
    init_env_state = jax.tree_util.tree_map(shard, init_env_state)

    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, action_dim))

    rng, actor_key, critic_key = jax.random.split(rng, 3)
    actor_params  = actor.init(actor_key,  dummy_obs)["params"]
    critic_params = critic.init(critic_key, dummy_obs, dummy_act)["params"]

    actor_state  = TrainState.create(
        apply_fn=actor.apply, params=actor_params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(policy_lr)),
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply, params=critic_params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(q_lr)),
    )

    log_alpha       = jnp.array(0.5)
    alpha_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(policy_lr))
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    rng, runner_rng = jax.random.split(rng)
    runner_rngs = jax.random.split(runner_rng, num_devices)

    runner_state = RunnerState(
        env_state=init_env_state,
        obs=init_obs,
        episode_return=jnp.zeros((num_devices, envs_per_device, num_agents)),
        actor_state=jax_utils.replicate(actor_state),
        critic_state=jax_utils.replicate(critic_state),
        target_critic_params=jax_utils.replicate(critic_params),
        log_alpha=jax_utils.replicate(log_alpha),
        alpha_opt_state=jax_utils.replicate(alpha_opt_state),
        buffer_state=jax_utils.replicate(buffer.init_state()),
        rng=runner_rngs,
        step_count=jax_utils.replicate(jnp.array(0)),
    )

    # ── Inner step function (PMAP'd) ──────────────────────────────────────────
    def _step_fn(runner_state: RunnerState, _):
        rng, action_key, step_key, sample_key, update_key, reset_key = \
            jax.random.split(runner_state.rng, 6)

        flat_obs = runner_state.obs.reshape(envs_per_device * num_agents, obs_dim)

        def explore_fn():
            return jax.random.uniform(
                action_key, shape=(envs_per_device * num_agents, action_dim),
                minval=-1.0, maxval=1.0,
            )

        def exploit_fn():
            action, _ = actor.apply(
                {"params": runner_state.actor_state.params},
                flat_obs, action_key, method=actor.get_action,
            )
            return action

        flat_action = jax.lax.cond(
            runner_state.step_count < learning_starts,
            explore_fn, exploit_fn,
        )
        action = flat_action.reshape(envs_per_device, num_agents, action_dim)

        step_keys = jax.random.split(step_key, envs_per_device)
        next_obs, next_env_state, reward, done, info = vmap_step(
            step_keys, runner_state.env_state, action, env_params
        )
        new_episode_return = runner_state.episode_return + reward

        flat_reward   = reward.flatten()
        flat_next_obs = next_obs.reshape(-1, obs_dim)
        flat_done     = done.flatten()

        new_buffer_state = buffer.add_batch(
            runner_state.buffer_state, flat_obs, flat_action, flat_reward, flat_next_obs, flat_done, total_insertions_per_step,
        )

        env_done   = jnp.any(done, axis=1)
        reset_keys = jax.random.split(reset_key, envs_per_device)
        reset_obs, reset_state = vmap_reset(reset_keys, env_params)

        final_obs            = jnp.where(env_done[:, None, None], reset_obs, next_obs)
        final_episode_return = jnp.where(env_done[:, None], 0.0, new_episode_return)

        def merge_states(reset_val, next_val):
            shape = (envs_per_device,) + (1,) * (next_val.ndim - 1)
            return jnp.where(jnp.reshape(env_done, shape), reset_val, next_val)

        final_env_state = jax.tree_util.tree_map(merge_states, reset_state, next_env_state)

        def perform_update():
            b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(
                new_buffer_state, sample_key, per_device_batch_size
            )
            batch = Transition(
                obs=b_obs, action=b_act, reward=b_rew, next_obs=b_next_obs, done=b_done,
            )
            key_critic, key_actor = jax.random.split(update_key)
            
            # Gradients are averaged across devices via pmean inside these update functions
            new_critic, q_loss = update_critic(
                runner_state.critic_state, runner_state.target_critic_params,
                runner_state.actor_state, runner_state.log_alpha, batch, gamma, key_critic,
            )
            new_actor, a_loss, log_prob = update_actor(
                runner_state.actor_state, new_critic,
                runner_state.log_alpha, batch.obs, key_actor,
            )
            new_log_alpha, new_alpha_opt, alpha_loss = update_alpha(
                runner_state.log_alpha, runner_state.alpha_opt_state,
                log_prob, target_entropy, alpha_optimizer,
            )
            new_target = jax.tree_util.tree_map(
                lambda t, c: tau_target * c + (1 - tau_target) * t,
                runner_state.target_critic_params, new_critic.params,
            )
            new_log_alpha = jnp.maximum(new_log_alpha, -2.0)
            return (new_actor, new_critic, new_target, new_log_alpha, new_alpha_opt, q_loss, a_loss)

        def skip_update():
            return (runner_state.actor_state, runner_state.critic_state,
                    runner_state.target_critic_params, runner_state.log_alpha,
                    runner_state.alpha_opt_state, 0.0, 0.0)

        (new_actor, new_critic, new_target, new_log_alpha,
         new_alpha_opt, q_loss, a_loss) = jax.lax.cond(
            runner_state.step_count >= learning_starts,
            perform_update, skip_update,
        )

        new_runner_state = runner_state.replace(
            env_state=final_env_state,
            obs=final_obs,
            episode_return=final_episode_return,
            actor_state=new_actor,
            critic_state=new_critic,
            target_critic_params=new_target,
            log_alpha=new_log_alpha,
            alpha_opt_state=new_alpha_opt,
            buffer_state=new_buffer_state,
            rng=rng,
            step_count=runner_state.step_count + 1,
        )

        metrics = {
            "reward":          jnp.mean(reward),
            "episode_return":  jnp.mean(new_episode_return),
            "env_done":        jnp.mean(env_done.astype(jnp.float32)),
            "q_loss":          q_loss,
            "a_loss":          a_loss,
            "alpha":           jnp.exp(new_log_alpha),
        }
        return new_runner_state, metrics

    # ── Compile ───────────────────────────────────────────────────────────────
    steps_per_epoch = 2_000
    num_epochs      = total_timesteps // steps_per_epoch

    console.print(f"[bold cyan]Compiling XLA Graph for Variant 5 across {num_devices} GPUs...[/bold cyan]")

    @functools.partial(jax.pmap, axis_name='devices')
    def run_epoch(runner_state):
        return jax.lax.scan(_step_fn, runner_state, None, length=steps_per_epoch)

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs("logs_film",         exist_ok=True)
    os.makedirs("checkpoints_film",  exist_ok=True)

    all_metrics = []
    start_time  = time.time()

    console.print(f"\n[bold green]Starting Variant 5 Distributed Training Loop ({num_devices} GPUs)...[/bold green]")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        runner_state, epoch_metrics = run_epoch(runner_state)
        # Block until ready on the first device's reward output to synchronize
        jax.block_until_ready(epoch_metrics["reward"][0])
        epoch_end = time.time()

        current_step = (epoch + 1) * steps_per_epoch
        
        # Average metrics across devices and sequence
        mean_reward  = float(jnp.mean(epoch_metrics["reward"]))
        mean_a_loss  = float(jnp.mean(epoch_metrics["a_loss"]))
        sps = (steps_per_epoch * num_envs * num_agents) / (epoch_end - epoch_start)

        console.print(
            f"Epoch {epoch+1:02d}/{num_epochs} | Step {current_step:>7,} | "
            f"Reward: {mean_reward:>7.2f} | Actor Loss: {mean_a_loss:>7.4f} | "
            f"Speed: {sps:,.0f} SPS"
        )
        
        # Collapse device dimension and sequence dimension for logging
        agg_metrics = {k: jnp.mean(v, axis=1) for k, v in epoch_metrics.items()} 
        # But wait, epoch_metrics is [num_devices, steps_per_epoch]
        # We can just average across devices, making it [steps_per_epoch]
        agg_metrics = {k: jnp.mean(v, axis=0) for k, v in epoch_metrics.items()}
        all_metrics.append(agg_metrics)

    end_time = time.time()
    duration = end_time - start_time
    total_sim = total_timesteps * num_envs * num_agents
    console.print(Panel.fit(
        f"[bold green]Distributed Training Complete![/bold green]\n"
        f"Simulated {total_sim:,} transitions in [bold white]{duration:.2f}s[/bold white]\n"
        f"Avg Speed: [bold magenta]{total_sim/duration:,.0f} SPS[/bold magenta]",
        border_style="green",
    ))

    # ── Save metrics + checkpoints ────────────────────────────────────────────
    metrics = {k: jnp.concatenate([m[k] for m in all_metrics]) for k in all_metrics[0]}

    metrics_df = pd.DataFrame({
        "step":                np.arange(total_timesteps),
        "mean_reward":         np.array(metrics["reward"]),
        "mean_episode_return": np.array(metrics["episode_return"]),
        "env_done_rate":       np.array(metrics["env_done"]),
        "q_loss":              np.array(metrics["q_loss"]),
        "a_loss":              np.array(metrics["a_loss"]),
        "alpha":               np.array(metrics["alpha"]),
    })
    metrics_df.to_csv("logs_film/metrics.csv", index=False)
    console.print("[bold green]✔ Metrics saved to logs_film/metrics.csv[/bold green]")

    ckpt = PyTreeCheckpointer()
    actor_path  = os.path.abspath("checkpoints_film/sac_actor_final")
    critic_path = os.path.abspath("checkpoints_film/sac_critic_final")
    
    # Unreplicate parameters from device 0 before saving
    final_actor_params = jax_utils.unreplicate(runner_state.actor_state.params)
    final_critic_params = jax_utils.unreplicate(runner_state.critic_state.params)
    
    ckpt.save(actor_path,  final_actor_params,  force=True)
    ckpt.save(critic_path, final_critic_params, force=True)
    console.print("[bold green]✔ Checkpoints saved to checkpoints_film/[/bold green]")


if __name__ == "__main__":
    main()
