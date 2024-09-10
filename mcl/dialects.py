from __future__ import annotations

import itertools


class LoopNestAPI:
    dim: tuple[int, ...]
    inner: LoopNestAPI | None

    def __init__(self, dim, inner=None):
        self.dim = dim
        self.inner = inner

    @staticmethod
    def from_tuple(args):
        def expand(*args):
            if not args:
                return None
            [head, *tail] = args
            return LoopNestAPI(head, inner=expand(*tail))
        return expand(*args)

    def __iter__(self):
        dims = self.get_dims()
        return iter(itertools.product(*map(lambda x: list(range(x)), dims)))

    def get_dims(self):
        if self.inner is None:
            return [self.dim]
        else:
            return [self.dim] + self.inner.get_dims()

    def reduce(self, fn, init):
        res = init
        for idx in self:
            res = fn(res, idx)
        return res


class ShapeAPI:
    dims: tuple[int,...]

    def __init__(self, dims):
        self.dims = tuple(dims)

    def deselect(self, axis):
        dims = self.dims
        return ShapeAPI([dims[i] for i in range(len(dims)) if i not in axis])

    def select(self, axis):
        dims = self.dims
        return ShapeAPI([dims[i] for i in range(len(dims)) if i in axis])

    def to_tuple(self):
        return tuple(self.dims)
