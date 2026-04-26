"""Acceptance tests for fizzbuzz. Pre-written. Do not modify.

Five tests covering:
  1. normal integer  → str(n)
  2. multiple of 3   → "Fizz"
  3. multiple of 5   → "Buzz"
  4. multiple of 15  → "FizzBuzz"
  5. n <= 0          → ValueError
"""
import pytest

from src.fizzbuzz import fizzbuzz


def test_returns_string_for_normal_number():
    assert fizzbuzz(1) == "1"
    assert fizzbuzz(2) == "2"
    assert fizzbuzz(7) == "7"


def test_multiple_of_three_returns_fizz():
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(9) == "Fizz"
    assert fizzbuzz(21) == "Fizz"


def test_multiple_of_five_returns_buzz():
    assert fizzbuzz(5) == "Buzz"
    assert fizzbuzz(10) == "Buzz"
    assert fizzbuzz(20) == "Buzz"


def test_multiple_of_fifteen_returns_fizzbuzz():
    assert fizzbuzz(15) == "FizzBuzz"
    assert fizzbuzz(30) == "FizzBuzz"
    assert fizzbuzz(45) == "FizzBuzz"


def test_zero_or_negative_raises_value_error():
    with pytest.raises(ValueError):
        fizzbuzz(0)
    with pytest.raises(ValueError):
        fizzbuzz(-1)
    with pytest.raises(ValueError):
        fizzbuzz(-15)
