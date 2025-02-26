from typing import Any
from utils import to_human_readable_name


class Pipeline:
    def __init__(self, steps: list) -> None:
        self.steps = steps

    def run(self) -> dict[str, Any]:
        data = {}
        for step in self.steps:
            print(f"Executing step {to_human_readable_name(step)}")
            data = step.run(data)
        return data

