"""Тесты к рубежной работе, вариант 1.

Использование в ноутбуке:
    from tests import test_task1
    test_task1(task1)

После успешного выполнения всех задач вызовите make_report().
"""

from __future__ import annotations

import builtins
import contextlib
import io
import sys
import time

import numpy as np
import pandas as pd

__all__ = ["test_task1", "test_task2", "test_task3", "make_report"]

_started_at = None
_results = {1: False, 2: False, 3: False}
_counts = {1: 10, 2: 10, 3: 10}


def _start_if_needed():
    global _started_at
    if _started_at is None:
        _started_at = time.monotonic()


def _elapsed():
    seconds = int(time.monotonic() - _started_at)
    return f"{seconds // 60} мин {seconds % 60} сек"


def _normal(text):
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _run_with_input(func, stdin_text):
    lines = iter(stdin_text.split("\n"))

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError("закончились входные данные") from None

    output = io.StringIO()
    old_input, old_stdin = builtins.input, sys.stdin
    builtins.input, sys.stdin = fake_input, io.StringIO(stdin_text + "\n")
    try:
        with contextlib.redirect_stdout(output):
            result = func()
        if result is not None and not output.getvalue().strip():
            return output.getvalue(), "функция должна печатать ответ через print()"
        return output.getvalue(), None
    except Exception as error:
        return output.getvalue(), f"{type(error).__name__}: {error}"
    finally:
        builtins.input, sys.stdin = old_input, old_stdin


def _after_test(number, passed):
    _results[number] = passed == _counts[number]
    print(f"Итог: {passed}/{_counts[number]}")
    if all(_results.values()):
        print(f"Все тесты пройдены. Прошло с первого запуска: {_elapsed()}.")


def _check_console(number, func, cases):
    _start_if_needed()
    if not callable(func):
        print("Передайте функцию без скобок: test_taskN(taskN)")
        _after_test(number, 0)
        return
    passed = 0
    for index, (stdin_text, expected) in enumerate(cases, 1):
        got, error = _run_with_input(func, stdin_text)
        if error:
            print(f"  Тест {index}: FAIL - {error}")
        elif _normal(got) == _normal(expected):
            passed += 1
            print(f"  Тест {index}: OK")
        else:
            print(f"  Тест {index}: FAIL")
            print(f"    ожидалось: {_normal(expected)!r}")
            print(f"    получено:  {_normal(got)!r}")
    _after_test(number, passed)


TASK1_CASES = [
    ("5\nann numpy 5\nbob python 2\nann numpy 8\nbob numpy 6\ncara python 9\n2", "numpy 2 8\npython 2 9"),
    ("0\n1", ""),
    ("4\nann ml 2\nbob ml 3\nann python 10\ncara ml 1\n2", "ml 3 3"),
    ("4\na z 5\nb z 7\na a 10\nb a 2\n1", "a 2 10\nz 2 7"),
    ("3\na x 9\na x 1\na x 5\n1", "x 1 9"),
    ("5\na x 1\nb y 2\nc z 3\nd y 4\ne x 5\n2", "x 2 5\ny 2 4"),
    ("4\na x 1\nb x 2\na x 3\nb x 4\n3", ""),
    ("6\na ml 1\nb ml 9\nc python 2\nd python 8\ne sql 4\na sql 7\n2", "ml 2 9\npython 2 8\nsql 2 7"),
    ("2\na data 100\nb data 1\n1", "data 2 100"),
    ("5\na a 1\nb b 2\nc c 3\nd d 4\ne e 5\n1", "a 1 1\nb 1 2\nc 1 3\nd 1 4\ne 1 5"),
]


TASK3_CASES = [
    ("5\n1 2 1 2 1\n3", "4"),
    ("3\n0 0 0\n0", "6"),
    ("5\n1 -1 1 -1 1\n0", "6"),
    ("4\n-2 2 -2 2\n0", "4"),
    ("1\n5\n5", "1"),
    ("4\n2 2 2 2\n4", "3"),
    ("4\n-1 -1 1 1\n0", "2"),
    ("5\n3 -2 2 -3 4\n1", "3"),
    ("6\n1 2 3 4 5 6\n100", "0"),
    ("5\n1 0 1 0 1\n1", "8"),
]


def test_task1(func):
    """Проверить функцию task1()."""
    print("Задача 1. Рейтинг тем")
    _check_console(1, func, TASK1_CASES)


def _expected_order_report(df):
    data = df.copy()
    data["net_revenue"] = data["qty"] * data["price"] * (1 - data["discount"].fillna(0))
    revenue = data.groupby("segment")["net_revenue"].sum()
    q1, q3 = np.percentile(data["net_revenue"], [25, 75])
    return {
        "top_segment": revenue.idxmax(),
        "above_median": sorted(revenue[revenue > revenue.median()].index.tolist()),
        "outliers": sorted(data.loc[data["net_revenue"] > q3 + 1.5 * (q3 - q1), "order_id"].tolist()),
    }


def _orders(order_ids, segments, qty, price, discount):
    return pd.DataFrame({"order_id": order_ids, "segment": segments, "qty": qty, "price": price, "discount": discount})


TASK2_CASES = [
    _orders([1, 2, 3, 4, 5, 6], ["A", "B", "A", "B", "A", "C"], [1, 2, 1, 1, 10, 1], [100, 100, 200, 300, 100, 100], [np.nan, .5, 0, 0, 0, 0]),
    _orders([10, 11, 12, 13], ["A", "B", "B", "C"], [1, 2, 1, 1], [50, 50, 120, 70], [0, .2, 0, np.nan]),
    _orders([20, 21], ["A", "A"], [1, 1], [10, 20], [0, 0]),
    _orders([30, 31, 32], ["C", "A", "B"], [2, 1, 3], [10, 30, 10], [np.nan, np.nan, np.nan]),
    _orders([40, 41, 42, 43], ["A", "B", "C", "D"], [1, 1, 1, 1], [5, 5, 5, 5], [0, 0, 0, 0]),
    _orders([50, 51, 52, 53, 54], ["A", "A", "B", "B", "C"], [1, 1, 1, 1, 1], [10, 20, 30, 40, 1000], [0, 0, 0, 0, 0]),
    _orders([60, 61, 62], ["X", "Y", "Y"], [3, 2, 4], [20, 30, 10], [.1, .25, 0]),
    _orders([70, 71, 72, 73], ["A", "B", "A", "C"], [1, 5, 2, 1], [100, 10, 80, 50], [0, .5, .25, 0]),
    _orders([80, 81, 82, 83, 84], ["M", "M", "N", "N", "N"], [1, 2, 3, 1, 1], [20, 30, 10, 40, 50], [0, 0, 0, .5, 0]),
    _orders([90, 91, 92, 93], ["A", "B", "C", "A"], [1, 1, 1, 1], [1, 2, 3, 100], [0, 0, 0, .99]),
]


def test_task2(func):
    """Проверить функцию task2(df)."""
    _start_if_needed()
    print("Задача 2. Отчёт по заказам")
    if not callable(func):
        print("Передайте функцию без скобок: test_task2(task2)")
        _after_test(2, 0)
        return
    passed = 0
    for index, source in enumerate(TASK2_CASES, 1):
        df = source.copy(deep=True)
        before = df.copy(deep=True)
        expected = _expected_order_report(df)
        try:
            actual = func(df)
            ok = actual == expected and df.equals(before)
        except Exception as error:
            ok, actual = False, f"{type(error).__name__}: {error}"
        if ok:
            passed += 1
            print(f"  Тест {index}: OK")
        else:
            print(f"  Тест {index}: FAIL")
            print(f"    ожидалось: {expected!r}")
            print(f"    получено:  {actual!r}")
            if not df.equals(before):
                print("    исходный DataFrame был изменён")
    _after_test(2, passed)


def test_task3(func):
    """Проверить функцию task3()."""
    print("Задача 3. Подотрезки с суммой")
    _check_console(3, func, TASK3_CASES)


def make_report():
    """Напечатать отчёт для сдачи, только если все 30 тестов пройдены."""
    if not all(_results.values()):
        print("Отчёт пока не готов: нужно пройти все тесты по трём задачам.")
        return
    print("ОТЧЁТ ПО РУБЕЖНОЙ РАБОТЕ")
    print("Вариант 1")
    for number in (1, 2, 3):
        print(f"Задача {number}: {_counts[number]}/{_counts[number]} тестов пройдено")
    print(f"Всего: {sum(_counts.values())}/{sum(_counts.values())}")
    print(f"Время с первого запуска теста: {_elapsed()}")
    print("Статус: ЗАЧЁТ")
