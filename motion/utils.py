import math
from functools import wraps
import torch


def smooth_01(t: torch.Tensor, inflection: float = 2.0) -> float:
    if isinstance(inflection, (int, float)):
        inflection = torch.tensor([inflection])
    error = torch.sigmoid(-inflection / 2)
    return torch.clamp((torch.sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0, 1)

def smooth(x0:float, x1:float, step:int, inflection: float = 2.0):
    t = torch.linspace(0, 1, steps=step)
    smooth_t = smooth_01(t, inflection=inflection)

    smooth_result = smooth_t * (x1-x0) + x0
    return smooth_result

if __name__ == "__main__":
    step = 10
    result1 = smooth(0, 3, step)
    result2 = smooth(1, 3.1111, step)
    result3 = smooth(1.11111, 3.1111, step)
    result4 = smooth(1.11111, 0.11, step)
    result5 = smooth(-1.11111, 0.11, step)
