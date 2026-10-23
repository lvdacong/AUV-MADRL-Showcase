from __future__ import annotations

import sys

from .goal_reach import train_main as goal_reach_train_main
from .goal_reach import viewer_main as goal_reach_viewer_main


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m auv_madrl "
            "{goal-reach-train|goal-reach-viewer} ..."
        )
    command = sys.argv.pop(1)
    if command == "goal-reach-train":
        goal_reach_train_main()
    elif command == "goal-reach-viewer":
        goal_reach_viewer_main()
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
