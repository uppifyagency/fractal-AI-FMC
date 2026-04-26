import numpy
import torch


DEFAULT_SEED = 160290


class TorchRandomState:
    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed(seed)

    def __getattr__(self, item):
        return getattr(torch, item)

    @staticmethod
    def seed(seed: int = DEFAULT_SEED):
        numpy.random.seed(seed)  # noqa: NPY002
        torch.random.manual_seed(int(seed))

    @staticmethod
    def permutation(x):
        idx = torch.randperm(x.shape[0])
        return x[idx]

    @staticmethod
    def random_sample(*args, **kwargs):
        return torch.rand(*args, **kwargs)

    @staticmethod
    def choice(a, size=None, replace=True, p=None):  # noqa: ARG004
        size = size if size is not None else 1
        if replace:
            size = size if isinstance(size, tuple) else (size,)
            indices = torch.randint(len(a), size)
            samples = a[indices]
        else:
            indices = torch.randperm(len(a))[:size]
            samples = a[indices]
        return samples

    @staticmethod
    def uniform(
        low=0.0,
        high=1.0,
        size=None,
        dtype=None,
    ):
        uniform = torch.distributions.uniform.Uniform(low, high)
        if size is not None:
            size = size if isinstance(size, tuple) else (size,)
            sample = uniform.sample(size)
        else:
            sample = uniform.sample()
        if dtype is not None:
            sample = sample.to(dtype)
        return sample

    @staticmethod
    def randint(low, high, size=None, dtype=None):
        size = size if size is not None else (1,)
        size = size if isinstance(size, tuple) else (size,)
        data = torch.randint(low, high, size)
        if dtype is not None:
            data = data.to(dtype)
        return data

    @staticmethod
    def normal(loc=0, scale=1.0, size=None):
        size = size if size is not None else (1,)
        size = size if isinstance(size, tuple) else (size,)
        return torch.normal(mean=loc, std=scale, size=size)

    @classmethod
    def random(cls, size=None):
        return cls.uniform(size=size)


random_state = TorchRandomState(seed=DEFAULT_SEED)
numpy_random_state = numpy.random.RandomState(seed=DEFAULT_SEED)
