with open('train_pure_jax.py', 'r') as f:
    content = f.read()

# Comment out restoration
content = content.replace(
    'actor_params = ckpt.restore(os.path.abspath("checkpoints_max/sac_actor_final"), item=actor_params)',
    '# actor_params = ckpt.restore(os.path.abspath("checkpoints_max/sac_actor_final"), item=actor_params)'
)
content = content.replace(
    'critic_params = ckpt.restore(os.path.abspath("checkpoints_max/sac_critic_final"), item=critic_params)',
    '# critic_params = ckpt.restore(os.path.abspath("checkpoints_max/sac_critic_final"), item=critic_params)'
)

# Change save directories
content = content.replace('checkpoints_spread', 'checkpoints_max_fresh')
content = content.replace('logs_spread', 'logs_max_fresh')

with open('train_pure_jax.py', 'w') as f:
    f.write(content)
