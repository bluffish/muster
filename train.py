"""Compatibility facade and entrypoint over :mod:`muster.rl`."""

from muster.rl.evaluation import AnchorEvaluator, write_evaluation_replay  # noqa: F401
from muster.rl.metrics import *  # noqa: F401,F403
from muster.rl.modes import *  # noqa: F401,F403
from muster.rl.pool import OpponentPool  # noqa: F401
from muster.rl.ppo import *  # noqa: F401,F403
from muster.rl.rollout import *  # noqa: F401,F403
from muster.rl.rollout import STATE_KEYS  # noqa: F401
from muster.rl.train import (  # noqa: F401
    compile_policies,
    main,
    parse_args,
    save_checkpoint,
    start_wandb,
)

if __name__ == "__main__":
    main()
