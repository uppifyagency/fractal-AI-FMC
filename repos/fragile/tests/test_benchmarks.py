import numpy as np
import pytest
import torch

from fragile.benchmarks import (
    Easom,
    easom,
    EggHolder,
    eggholder,
    holder_table,
    HolderTable,
    lennard_jones,
    LennardJones,
    Rastrigin,
    rastrigin,
    Rosenbrock,
    rosenbrock,
    Sphere,
    sphere,
    styblinski_tang,
    StyblinskiTang,
)


@pytest.fixture
def test_inputs():
    """
    Fixture providing test inputs for the benchmark functions.
    """
    return {
        "sphere": torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, -2.0]]),
        "rastrigin": torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, -2.0]]),
        "eggholder": torch.tensor([[512.0, 404.2319], [0.0, 0.0], [100.0, -100.0]]),
        "styblinski_tang": torch.tensor([[2.903534, 2.903534], [-2.903534, -2.903534]]),
        "rosenbrock": torch.tensor([[1.0, 1.0], [1.2, 1.44], [0.0, 0.0]]),
        "easom": torch.tensor([[np.pi, np.pi], [0.0, 0.0]]),
        "holder_table": torch.tensor([[8.05502, 9.66459], [0.0, 0.0]]),
        "lennard_jones": torch.rand(2, 6) * 10 - 5,  # Random positions for 2 atoms
    }


# Test the benchmark functions directly
def test_sphere(test_inputs):
    x = test_inputs["sphere"]
    expected_values = torch.sum(x**2, dim=1)
    output = sphere(x)
    assert torch.allclose(output, expected_values), "Sphere function output mismatch"


def test_rastrigin(test_inputs):
    x = test_inputs["rastrigin"]
    dims = x.shape[1]
    a = 10
    expected_values = a * dims + torch.sum(x**2 - a * torch.cos(2 * np.pi * x), dim=1)
    output = rastrigin(x)
    assert torch.allclose(output, expected_values), "Rastrigin function output mismatch"


def test_eggholder(test_inputs):
    x = test_inputs["eggholder"]
    output = eggholder(x)
    # Known global minimum
    expected_min = -959.6407
    assert torch.isclose(output[0], torch.tensor(expected_min), atol=1e-3), (
        "Eggholder function minimum value mismatch"
    )


def test_styblinski_tang(test_inputs):
    x = test_inputs["styblinski_tang"]
    output = styblinski_tang(x)
    expected_value = torch.tensor([-49.2970, -78.3324])  # Known minimum value per sample
    assert torch.allclose(output, expected_value, atol=1e-3), (
        "Styblinski-Tang function output mismatch"
    )


def test_rosenbrock(test_inputs):
    x = test_inputs["rosenbrock"]
    output = rosenbrock(x)
    expected_values = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(output, expected_values, atol=1e-3), (
        "Rosenbrock function output mismatch"
    )


def test_easom(test_inputs):
    x = test_inputs["easom"]
    output = easom(x)
    expected_values = torch.tensor([-1.0, -0.0])
    assert torch.allclose(output, expected_values, atol=1e-3), "Easom function output mismatch"


def test_holder_table(test_inputs):
    x = test_inputs["holder_table"]
    output = holder_table(x)
    expected_values = torch.tensor([-19.2085, -0.0])
    assert torch.allclose(output, expected_values, atol=1e-3), (
        "Holder Table function output mismatch"
    )


def test_lennard_jones(test_inputs):
    x = test_inputs["lennard_jones"]
    output = lennard_jones(x)
    assert output.shape[0] == x.shape[0], "Lennard-Jones function output shape mismatch"


# Test the OptimBenchmark classes
def test_sphere_class():
    dims = 2
    benchmark = Sphere(dims)
    x = torch.zeros((1, dims))
    output = benchmark(x)
    expected_value = torch.tensor([0.0])
    assert torch.allclose(output, expected_value), "Sphere class output mismatch"


def test_rastrigin_class():
    dims = 2
    benchmark = Rastrigin(dims)
    x = torch.zeros((1, dims))
    output = benchmark(x)
    expected_value = torch.tensor([0.0])
    assert torch.allclose(output, expected_value), "Rastrigin class output mismatch"


def test_eggholder_class():
    benchmark = EggHolder()
    x = torch.tensor([[512.0, 404.2319]])
    output = benchmark(x)
    expected_value = torch.tensor([-959.6407])
    assert torch.allclose(output, expected_value, atol=1e-3), "EggHolder class output mismatch"


def test_styblinski_tang_class():
    dims = 2
    benchmark = StyblinskiTang(dims)
    x = torch.ones((1, dims)) * -2.903534
    output = benchmark(x)
    expected_value = torch.tensor([-78.3324])
    assert torch.allclose(output, expected_value, atol=1e-3), (
        "Styblinski-Tang class output mismatch"
    )


def test_rosenbrock_class():
    dims = 2
    benchmark = Rosenbrock(dims)
    x = torch.ones((1, dims))
    output = benchmark(x)
    expected_value = torch.tensor([0.0])
    assert torch.allclose(output, expected_value, atol=1e-3), "Rosenbrock class output mismatch"


def test_easom_class():
    benchmark = Easom()
    x = torch.ones((1, 2)) * np.pi
    output = benchmark(x)
    expected_value = torch.tensor([-1.0])
    assert torch.allclose(output, expected_value, atol=1e-3), "Easom class output mismatch"


def test_holder_table_class():
    benchmark = HolderTable()
    x = torch.tensor([[8.05502, 9.66459]])
    output = benchmark(x)
    expected_value = torch.tensor([-19.2085])
    assert torch.allclose(output, expected_value, atol=1e-3), "HolderTable class output mismatch"


def test_lennard_jones_class():
    n_atoms = 3
    benchmark = LennardJones(n_atoms)
    x = torch.rand((1, n_atoms * 3)) * 10 - 5  # Random initial positions
    output = benchmark(x)
    assert output.shape[0] == x.shape[0], "Lennard-Jones class output shape mismatch"


# Test the sample method in the OptimBenchmark classes
def test_benchmark_sampling():
    benchmarks = [
        Sphere(2),
        Rastrigin(2),
        EggHolder(),
        StyblinskiTang(2),
        Rosenbrock(2),
        Easom(),
        HolderTable(),
        LennardJones(3),
    ]
    for benchmark in benchmarks:
        samples = benchmark.sample(10)
        assert samples.shape[0] == 10, f"Sample size mismatch in {benchmark.__class__.__name__}"
        assert samples.shape[1] == benchmark.dims, (
            f"Sample dimension mismatch in {benchmark.__class__.__name__}"
        )


# Test the bounds in the OptimBenchmark classes
def test_benchmark_bounds():
    benchmarks = [
        Sphere(2),
        Rastrigin(2),
        EggHolder(),
        StyblinskiTang(2),
        Rosenbrock(2),
        Easom(),
        HolderTable(),
        LennardJones(3),
    ]
    for benchmark in benchmarks:
        bounds = benchmark.bounds
        assert hasattr(bounds, "low") and hasattr(bounds, "high"), (
            f"Bounds missing in {benchmark.__class__.__name__}"
        )
        assert bounds.shape == (benchmark.dims,), (
            f"Bounds shape mismatch in {benchmark.__class__.__name__}"
        )

        # Test that the best_state is within bounds
        best_state = benchmark.best_state
        if best_state is None:
            return
        if isinstance(best_state, torch.Tensor):
            contains = bounds.contains(best_state.unsqueeze(0)).all()
        else:
            contains = bounds.contains(torch.tensor(best_state).unsqueeze(0)).all()
        assert contains, f"Best state out of bounds in {benchmark.__class__.__name__}"


# Test the benchmark property in the OptimBenchmark classes
def test_benchmark_value():
    benchmarks = [
        Sphere(2),
        Rastrigin(2),
        EggHolder(),
        StyblinskiTang(2),
        Rosenbrock(2),
        Easom(),
        HolderTable(),
    ]
    for benchmark in benchmarks:
        x = benchmark.best_state.unsqueeze(0)
        output = benchmark(x)
        expected_value = benchmark.benchmark
        if expected_value is not None:
            if isinstance(expected_value, torch.Tensor):
                expected_value = expected_value.expand_as(output)
            assert torch.allclose(
                output.to(torch.float64), expected_value.to(torch.float64), atol=1e-3
            ), f"Benchmark value mismatch in {benchmark.__class__.__name__}"
