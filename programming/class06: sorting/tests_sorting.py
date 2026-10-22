"""Автотесты к ноутбуку «Сортировка в Python».

В каждой задаче напишите функцию ``taskN()``, которая читает вход через
``input()`` и печатает только ответ. Пример использования в ноутбуке::

    from tests_sorting import test_task1

    def task1():
        n = int(input())
        numbers = list(map(int, input().split()))
        k, x = map(int, input().split())
        print(*sorted(numbers, key=lambda value: (abs(value - x), value))[:k])

    test_task1(task1)

Для запуска своего примера: ``run_with_input(task1, "3\\n3 1 2\\n2 2")``.
"""

import builtins
import contextlib
import io
import sys
import traceback

__all__ = ["run_with_input"] + [f"test_task{i}" for i in range(1, 16)]


def _run(func, stdin_text):
    """Вернуть напечатанный текст и сообщение об ошибке, если она была."""
    lines = iter(stdin_text.split("\n"))

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError("входные данные закончились") from None

    output = io.StringIO()
    old_input, old_stdin = builtins.input, sys.stdin
    builtins.input, sys.stdin = fake_input, io.StringIO(stdin_text + "\n")
    try:
        with contextlib.redirect_stdout(output):
            result = func()
        if result is not None and not output.getvalue().strip():
            return output.getvalue(), "функция вернула ответ через return, а нужно print()"
        return output.getvalue(), None
    except Exception:
        return output.getvalue(), traceback.format_exc(limit=1).strip().splitlines()[-1]
    finally:
        builtins.input, sys.stdin = old_input, old_stdin


def run_with_input(func, stdin_text):
    """Запустить функцию на своём вводе и вернуть её вывод."""
    output, error = _run(func, stdin_text)
    if error:
        raise RuntimeError(error)
    return output


def _normal(text):
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _show(label, value):
    print(f"     {label}: {(value or '(пусто)').replace(chr(10), ' | ')}")


def _check(number, func):
    task = _TASKS[number]
    if not callable(func):
        print(f"❌ test_task{number}: передайте функцию без скобок: test_task{number}(task{number})")
        return
    print(f"Задача {number}. {task['title']} — тестов: {len(task['cases'])}")
    passed = 0
    for index, (stdin_text, expected) in enumerate(task["cases"], 1):
        got, error = _run(func, stdin_text)
        if error:
            print(f"  ❌ тест {index}: ошибка — {error}")
            _show("ввод", stdin_text)
        elif _normal(got) == _normal(expected):
            passed += 1
            print(f"  ✅ тест {index}: OK")
        else:
            print(f"  ❌ тест {index}: неверный ответ")
            _show("ввод", stdin_text)
            _show("ожидалось", expected)
            _show("получено", got)
    mark = "✅" if passed == len(task["cases"]) else "❌"
    print(f"Итог: пройдено {passed} из {len(task['cases'])} {mark}")


_TASKS = {
    1: {"title": "K ближайших чисел", "cases": [
        ("6\n1 8 3 10 6 4\n4 5", "4 6 3 8"),
        ("5\n-5 -1 1 5 9\n3 0", "-1 1 -5"),
        ("4\n2 2 2 2\n2 100", "2 2"),
    ]},
    2: {"title": "Таблица турнира", "cases": [
        ("4\nAnn 3 20 100\nBob 4 50 100\nCara 4 30 120\nDan 4 30 120", "Cara\nDan\nBob\nAnn"),
        ("3\nZoey 1 0 10\nAmy 1 0 10\nMax 2 99 9", "Amy\nZoey\nMax"),
    ]},
    3: {"title": "Хронология дат", "cases": [
        ("4\n01.01.2025\n31.12.2024\n02.01.2025\n01.01.2024", "01.01.2024\n31.12.2024\n01.01.2025\n02.01.2025"),
        ("3\n29.02.2024\n28.02.2024\n01.03.2024", "28.02.2024\n29.02.2024\n01.03.2024"),
    ]},
    4: {"title": "Ближайшая пара", "cases": [
        ("6\n8 1 5 12 3 10", "1 3"),
        ("5\n-10 -3 -1 4 9", "-3 -1"),
        ("4\n7 2 7 9", "7 7"),
    ]},
    5: {"title": "Минимальный разброс", "cases": [
        ("7 3\n10 100 300 200 1000 20 30", "20"),
        ("5 1\n5 1 9 3 7", "0"),
        ("5 4\n1 9 3 20 12", "11"),
    ]},
    6: {"title": "Третье различное", "cases": [
        ("6\n2 2 3 1 5 5", "2"),
        ("3\n1 1 2", "2"),
        ("5\n-1 -2 -3 -4 -5", "-3"),
    ]},
    7: {"title": "Пары с заданной разностью", "cases": [
        ("6\n1 5 3 4 2 3\n2", "5"),
        ("4\n1 1 1 1\n0", "6"),
        ("5\n-3 -1 0 2 4\n3", "2"),
    ]},
    8: {"title": "Три числа ближе всего к цели", "cases": [
        ("4\n-1 2 1 -4\n1", "2"),
        ("5\n0 0 0 0 0\n1", "0"),
        ("5\n-8 -2 1 4 9\n3", "3"),
    ]},
    9: {"title": "Минимум обменов", "cases": [
        ("4\n4 3 2 1", "2"),
        ("5\n10 20 30 40 50", "0"),
        ("5\n2 5 3 1 4", "3"),
    ]},
    10: {"title": "Объединение отрезков", "cases": [
        ("4\n1 3\n2 6\n8 10\n9 12", "1 6\n8 12"),
        ("3\n5 7\n1 2\n3 4", "1 2\n3 4\n5 7"),
        ("3\n1 4\n4 5\n2 3", "1 5"),
    ]},
    11: {"title": "Максимум мероприятий", "cases": [
        ("4\n1 3\n2 5\n4 6\n6 7", "3"),
        ("3\n1 2\n2 3\n3 4", "3"),
        ("3\n1 10\n2 3\n4 5", "2"),
    ]},
    12: {"title": "Сколько нужно аудиторий", "cases": [
        ("4\n1 4\n2 5\n3 6\n7 8", "3"),
        ("3\n1 2\n2 3\n3 4", "1"),
        ("4\n1 10\n2 3\n4 5\n6 7", "2"),
    ]},
    13: {"title": "Минимум стрел", "cases": [
        ("4\n10 16\n2 8\n1 6\n7 12", "2"),
        ("3\n1 2\n3 4\n5 6", "3"),
        ("3\n1 10\n2 9\n3 8", "1"),
    ]},
    14: {"title": "Максимум одновременных поездов", "cases": [
        ("4\n1 5\n2 6\n5 7\n5 8", "4"),
        ("3\n1 2\n2 3\n3 4", "2"),
        ("3\n1 10\n2 3\n4 5", "2"),
    ]},
    15: {"title": "Количество инверсий", "cases": [
        ("5\n2 4 1 3 5", "3"),
        ("4\n4 3 2 1", "6"),
        ("5\n1 1 1 1 1", "0"),
    ]},
}


def _make_test(number):
    def test(func):
        _check(number, func)
    test.__name__ = f"test_task{number}"
    test.__doc__ = f"Проверить решение задачи {number}."
    return test


globals().update({f"test_task{i}": _make_test(i) for i in range(1, 16)})
