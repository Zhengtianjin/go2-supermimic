"""Registers the chase / dodge tasks with mjlab's task registry on import."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from go2_supermimic.env.env_cfg import go2_chase_env_cfg

for task_id, kw in (
  ("Go2-Chase-Flat", {}),
  ("Go2-Chase-Obstacles", {"obstacles": True}),
  ("Go2-Chase-Terrain", {"terrain": True}),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=go2_chase_env_cfg(**kw),
    play_env_cfg=go2_chase_env_cfg(play=True, **kw),
    rl_cfg=RslRlOnPolicyRunnerCfg(),  # unused: training is SuperMimic (scripts/train.py)
  )
