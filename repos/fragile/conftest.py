import numpy
import plangym
import pytest

import fragile


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["plangym"] = plangym
    doctest_namespace["fragile"] = fragile
