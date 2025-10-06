"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable




def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / x


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    return d * (-1.0 / (x ** 2))


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0





# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# ## Task 0.3

def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    it = iter(ls)
    try:
        acc = next(it)
    except StopIteration:
        raise ValueError("reduce() of empty sequence with no initial value")
    for x in it:
        acc = fn(acc, x)
    return acc



def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    seq = list(ls)
    if not seq:
        return 0.0
    return reduce(add, seq)


def prod(ls: Iterable[float]) -> float:
    seq = list(ls)
    if not seq:
        return 1.0
    return reduce(mul, seq)
