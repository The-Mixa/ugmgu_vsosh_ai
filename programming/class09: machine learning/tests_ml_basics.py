"""Автотесты к уроку «Линейная и логистическая регрессия».

Использование:

    from tests_ml_basics import test_task1
    test_task1(mse)
"""

import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import log_loss

__all__ = [f"test_task{i}" for i in range(1, 11)]


def _check(number, name, func, check):
    if not callable(func):
        print(f"❌ test_task{number}: передайте функцию без скобок")
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            check(func)
    except Exception as exc:
        print(f"Задача {number}. {name}: ❌ {type(exc).__name__}: {exc}")
    else:
        print(f"Задача {number}. {name}: все проверки пройдены ✅")


def _assert_array(got, expected):
    assert isinstance(got, np.ndarray), "ожидался массив NumPy"
    assert got.shape == expected.shape, f"неверная форма: {got.shape}, ожидалось {expected.shape}"
    assert np.allclose(got, expected, atol=1e-6), f"ожидалось {expected}, получено {got}"


def test_task1(func):
    def check(fn):
        assert np.isclose(fn(np.array([1, 3]), np.array([2, 1])), 2.5)
        assert np.isclose(fn(np.array([4, 4]), np.array([4, 4])), 0.0)
    _check(1, "mse", func, check)


def test_task2(func):
    def check(fn):
        _assert_array(fn(np.array([1, 2, 3]), 2, 1), np.array([3, 5, 7]))
        _assert_array(fn(np.array([-1, 0]), -3, 2), np.array([5, 2]))
    _check(2, "linear_predict", func, check)


def test_task3(func):
    def check(fn):
        X = np.array([[1.], [2.], [3.], [4.]])
        y = np.array([3., 5., 7., 9.])
        model = fn(X, y)
        assert isinstance(model, LinearRegression), "верните LinearRegression"
        assert np.allclose(model.predict([[5.]]), [11.])
    _check(3, "fit_linear", func, check)


def test_task4(func):
    def check(fn):
        X = np.array([[1.], [2.], [3.], [4.]])
        y = np.array([3., 5., 7., 9.])
        model = fn(X, y, 0.5)
        assert isinstance(model, Ridge), "верните Ridge"
        assert np.isclose(model.alpha, 0.5), "передайте alpha в Ridge(alpha=alpha)"
        assert model.predict([[5.]])[0] > 9
    _check(4, "fit_ridge", func, check)


def test_task5(func):
    def check(fn):
        _assert_array(fn(np.array([-2., 0., 2.])), np.array([0.11920292, 0.5, 0.88079708]))
    _check(5, "sigmoid", func, check)


def test_task6(func):
    def check(fn):
        _assert_array(fn(np.array([0.1, 0.5, 0.9]), 0.5), np.array([0, 1, 1]))
        _assert_array(fn(np.array([0.1, 0.5, 0.9]), 0.8), np.array([0, 0, 1]))
    _check(6, "classify_probabilities", func, check)


def test_task7(func):
    def check(fn):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.8, 0.7, 0.2])
        assert np.isclose(fn(y, p), log_loss(y, p))
    _check(7, "binary_log_loss", func, check)


def test_task8(func):
    def check(fn):
        train = pd.DataFrame({"a": [1., 2., 3.], "b": [10., 20., 30.]})
        test = pd.DataFrame({"a": [4.], "b": [40.]})
        got = fn(train, test)
        assert isinstance(got, tuple) and len(got) == 2, "верните кортеж (train_scaled, test_scaled)"
        train_scaled, test_scaled = got
        assert isinstance(train_scaled, pd.DataFrame) and isinstance(test_scaled, pd.DataFrame)
        assert np.allclose(train_scaled.mean().values, [0, 0])
        assert np.allclose(train_scaled.std(ddof=0).values, [1, 1])
        assert list(test_scaled.columns) == ["a", "b"]
    _check(8, "standardize_train_test", func, check)


def test_task9(func):
    def check(fn):
        X = np.array([[0.], [0.2], [0.8], [1.], [1.2], [1.4]])
        y = np.array([0, 0, 1, 1, 1, 1])
        model = fn(X, y, 0.7)
        assert isinstance(model, LogisticRegression), "верните LogisticRegression"
        assert np.isclose(model.C, 0.7), "передайте C в LogisticRegression(C=C)"
        assert model.predict([[1.4]])[0] == 1, "модель должна быть обучена через fit"
    _check(9, "fit_logistic", func, check)


def test_task10(func):
    def check(fn):
        X = np.array([[0.], [0.2], [0.8], [1.], [1.2], [1.4]])
        y = np.array([0, 0, 1, 1, 1, 1])
        model = fn(X, y, 1.0)
        assert isinstance(model, LogisticRegression), "верните LogisticRegression"
        assert np.isclose(model.l1_ratio, 1.0), "укажите l1_ratio=1 для L1-регуляризации"
        assert model.solver == "saga", "для l1_ratio=1 используйте solver='saga'"
        assert np.array_equal(model.predict([[0.], [1.]]), [0, 1])
    _check(10, "fit_l1_logistic", func, check)
