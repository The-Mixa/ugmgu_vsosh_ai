"""Автотесты к уроку «Основы машинного обучения».

Импортируйте нужный тест рядом с решением, например::

    from tests_ml_basics import test_task1
    test_task1(select_columns)

Каждый тест передаёт небольшие искусственные данные. Поэтому решения не должны
зависеть от конкретных строк Titanic и не должны изменять входную таблицу.
"""

import contextlib
import io
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

__all__ = [f"test_task{i}" for i in range(1, 11)]


def _df():
    return pd.DataFrame({
        "survived": [0, 1, 1, 0, 1, 0],
        "pclass": [3, 1, 2, 3, 1, 2],
        "sex": ["male", "female", "female", "male", "female", "male"],
        "age": [22.0, np.nan, 28.0, 35.0, 18.0, 40.0],
        "fare": [7.25, 71.28, 13.0, 8.05, 53.1, 20.0],
        "embarked": ["S", "C", "S", "S", "C", "Q"],
    })


def _equal(got, expected):
    if isinstance(expected, pd.DataFrame):
        if not isinstance(got, pd.DataFrame):
            return False, "ожидался DataFrame"
        try:
            pd.testing.assert_frame_equal(got, expected, check_dtype=False, check_like=False)
            return True, ""
        except AssertionError as exc:
            return False, str(exc).split("\n")[0]
    if isinstance(expected, pd.Series):
        if not isinstance(got, pd.Series):
            return False, "ожидался Series"
        try:
            pd.testing.assert_series_equal(got, expected, check_dtype=False, check_names=False)
            return True, ""
        except AssertionError as exc:
            return False, str(exc).split("\n")[0]
    if isinstance(expected, tuple):
        return (isinstance(got, tuple) and len(got) == len(expected)
                and all(_equal(x, y)[0] for x, y in zip(got, expected))), ""
    if isinstance(expected, np.ndarray):
        if not isinstance(got, np.ndarray) or got.shape != expected.shape:
            return False, f"ожидался массив формы {expected.shape}"
        return bool(np.allclose(got, expected, equal_nan=True)), "значения не совпали"
    if isinstance(expected, (float, int, np.floating, np.integer)):
        return bool(np.isclose(got, expected)), f"ожидалось {expected}, получено {got}"
    return got == expected, f"ожидалось {expected}, получено {got}"


def _call(fn, *args):
    before = [x.copy(deep=True) if isinstance(x, pd.DataFrame) else x.copy() if isinstance(x, np.ndarray) else None for x in args]
    with contextlib.redirect_stdout(io.StringIO()):
        result = fn(*args)
    for value, snapshot in zip(args, before):
        if isinstance(value, pd.DataFrame) and not value.equals(snapshot):
            raise AssertionError("функция изменила исходный DataFrame")
        if isinstance(value, np.ndarray) and not np.array_equal(value, snapshot):
            raise AssertionError("функция изменила исходный массив")
    return result


def _run(number, name, fn, cases):
    if not callable(fn):
        print(f"❌ test_task{number}: передайте функцию без круглых скобок")
        return
    print(f"Задача {number}. {name} — тестов: {len(cases)}")
    passed = 0
    for i, (args, expected) in enumerate(cases, 1):
        try:
            got = _call(fn, *args)
            ok, hint = _equal(got, expected)
        except Exception as exc:
            ok, hint = False, f"{type(exc).__name__}: {exc}"
        if ok:
            passed += 1
            print(f"  ✅ тест {i}: OK")
        else:
            print(f"  ❌ тест {i}: {hint}")
    print(f"Итог: пройдено {passed} из {len(cases)}" + (" ✅" if passed == len(cases) else " ❌"))


def test_task1(fn):
    _run(1, "standardize", fn, [
        ((np.array([1.0, 2.0, 3.0]),), np.array([-1.22474487, 0.0, 1.22474487])),
        ((np.array([5.0, 5.0, 5.0]),), np.array([0.0, 0.0, 0.0])),
    ])


def test_task2(fn):
    data = _df()
    _run(2, "select_columns", fn, [( (data,), data[["age", "fare", "survived"]] ),
                                    ( (data.iloc[:0],), data.iloc[:0][["age", "fare", "survived"]] )])


def test_task3(fn):
    data = _df(); expected = data.copy(); expected["age"] = expected["age"].fillna(expected["age"].median())
    one_missing = pd.DataFrame({"age": [np.nan, 10.0, 30.0], "fare": [1, 2, 3]})
    expected2 = one_missing.copy(); expected2["age"] = [20.0, 10.0, 30.0]
    _run(3, "fill_age", fn, [((data,), expected), ((one_missing,), expected2)])


def test_task4(fn):
    data = _df(); expected = data.copy(); expected["sex_code"] = [0, 1, 1, 0, 1, 0]
    other = pd.DataFrame({"sex": ["female", "male", "female"]})
    expected2 = other.copy(); expected2["sex_code"] = [1, 0, 1]
    _run(4, "encode_sex", fn, [((data,), expected), ((other,), expected2)])


def test_task5(fn):
    data = _df()
    _run(5, "survival_rate_by_group", fn, [((data, "sex"), data.groupby("sex")["survived"].mean()),
                                             ((data, "pclass"), data.groupby("pclass")["survived"].mean())])


def test_task6(fn):
    data = _df()
    prepared = data.copy(); prepared["age"] = prepared["age"].fillna(prepared["age"].median()); prepared["sex_code"] = (prepared["sex"] == "female").astype(int)
    expected = prepared[["pclass", "age", "fare", "sex_code"]]
    small = data.iloc[:3]
    small_prepared = small.copy(); small_prepared["age"] = small_prepared["age"].fillna(small_prepared["age"].median()); small_prepared["sex_code"] = (small_prepared["sex"] == "female").astype(int)
    expected_small = small_prepared[["pclass", "age", "fare", "sex_code"]]
    _run(6, "make_features", fn, [((data,), expected), ((small,), expected_small)])


def test_task7(fn):
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    try:
        model = _call(fn, x, y)
        ok = isinstance(model, LinearRegression) and np.allclose(model.predict([[5.0]]), [11.0])
        _run(7, "fit_linear", lambda *_: ok, [((None,), True)])
    except Exception as exc:
        _run(7, "fit_linear", lambda *_: (_ for _ in ()).throw(exc), [((None,), True)])


def test_task8(fn):
    x = np.array([[0], [0.2], [0.8], [1.0], [1.2], [1.4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    try:
        model = _call(fn, x, y)
        ok = isinstance(model, LogisticRegression) and np.array_equal(model.predict([[0], [1]]), [0, 1])
        _run(8, "fit_logistic", lambda *_: ok, [((None,), True)])
    except Exception as exc:
        _run(8, "fit_logistic", lambda *_: (_ for _ in ()).throw(exc), [((None,), True)])


def test_task9(fn):
    x = np.array([[0], [0.2], [0.8], [1.0], [1.2], [1.4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    try:
        model = _call(fn, x, y)
        ok = isinstance(model, DecisionTreeClassifier) and model.get_params()["max_depth"] == 2 and accuracy_score(y, model.predict(x)) == 1
        _run(9, "fit_tree", lambda *_: ok, [((None,), True)])
    except Exception as exc:
        _run(9, "fit_tree", lambda *_: (_ for _ in ()).throw(exc), [((None,), True)])


def test_task10(fn):
    x = np.array([[0], [0.2], [0.8], [1.0], [1.2], [1.4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    try:
        model = _call(fn, x, y)
        ok = isinstance(model, RandomForestClassifier) and model.get_params()["n_estimators"] >= 50 and np.array_equal(model.predict([[0], [1]]), [0, 1])
        _run(10, "fit_forest", lambda *_: ok, [((None,), True)])
    except Exception as exc:
        _run(10, "fit_forest", lambda *_: (_ for _ in ()).throw(exc), [((None,), True)])
