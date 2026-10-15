"""Автотесты к ноутбуку «Основы алгоритмов: линейные приёмы».

Каждая задача в ноутбуке оформлена как ``taskN()``: функция сама читает
вход через input() и печатает ответ. Например::

    from tests_algorithms import test_task1

    def task1():
        n = int(input())
        a = list(map(int, input().split()))
        print(max(a))

    test_task1(task1)

Для собственного примера используйте ``run_with_input(task1, "3\\n1 9 2")``.
Лишние пробелы в конце строк вывода не учитываются.
"""

import builtins
import contextlib
import io
import sys
import traceback

__all__ = ["run_with_input"] + [f"test_task{i}" for i in range(1, 31)]


def _run(func, stdin_text):
    """Вернуть (всё_напечатанное, исключение_или_None) после запуска func."""
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
            return output.getvalue(), "Функция вернула ответ через return, а нужно print()."
        return output.getvalue(), None
    except Exception:
        return output.getvalue(), traceback.format_exc(limit=1).strip().splitlines()[-1]
    finally:
        builtins.input, sys.stdin = old_input, old_stdin


def run_with_input(func, stdin_text):
    """Запустить решение на своём вводе и вернуть его текстовый вывод."""
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
    value = value if value else "(пусто)"
    print(f"     {label}: {value.replace(chr(10), ' | ')}")


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
    1: {"title": "Максимум за один проход", "cases": [
        ("5\n3 1 9 2 7", "9"), ("1\n-5", "-5"), ("4\n0 0 0 0", "0")]},
    2: {"title": "Индекс первого максимума", "cases": [
        ("5\n3 9 1 9 2", "1"), ("3\n-1 -2 -3", "0"), ("1\n42", "0")]},
    3: {"title": "Количество соседних подъёмов", "cases": [
        ("5\n1 3 2 2 5", "2"), ("1\n10", "0"), ("4\n4 3 2 1", "0")]},
    4: {"title": "Самая длинная серия одинаковых", "cases": [
        ("8\n1 1 2 2 2 3 3 1", "3"), ("1\n7", "1"), ("5\n0 0 0 0 0", "5")]},
    5: {"title": "Префиксные суммы", "cases": [
        ("5\n2 -1 3 0 4", "2 1 4 4 8"), ("3\n-1 -2 -3", "-1 -3 -6")]},
    6: {"title": "Сумма на отрезке", "cases": [
        ("5\n2 -1 3 0 4\n1 3", "2"), ("3\n5 6 7\n0 2", "18"), ("1\n9\n0 0", "9")]},
    7: {"title": "Равновесный индекс", "cases": [
        ("5\n1 3 5 2 2", "2"), ("3\n1 2 3", "-1"), ("1\n0", "0")]},
    8: {"title": "Слияние двух отсортированных списков", "cases": [
        ("3\n1 4 8\n4\n2 3 7 9", "1 2 3 4 7 8 9"), ("0\n\n3\n1 1 1", "1 1 1")]},
    9: {"title": "Пара с заданной суммой", "cases": [
        ("5\n1 2 4 7 11\n9", "YES"), ("4\n1 3 5 8\n20", "NO"), ("4\n-3 -1 0 4\n1", "YES")]},
    10: {"title": "Удалить дубликаты из отсортированного", "cases": [
        ("7\n1 1 2 2 2 4 5", "1 2 4 5"), ("1\n8", "8"), ("5\n3 3 3 3 3", "3")]},
    11: {"title": "Палиндром без пробелов", "cases": [
        ("А роза упала на лапу Азора", "YES"), ("python", "NO"), ("Шалаш", "YES")]},
    12: {"title": "Самый длинный отрезок с суммой не больше K", "cases": [
        ("6\n1 2 1 1 1 3\n4", "4"), ("3\n5 1 1\n3", "2"), ("4\n1 1 1 1\n10", "4")]},
    13: {"title": "Сколько подотрезков имеют сумму S", "cases": [
        ("5\n1 2 1 2 1\n3", "4"), ("4\n0 0 0 0\n0", "10"), ("3\n1 1 1\n5", "0")]},
    14: {"title": "Ближайшая пара между двумя списками", "cases": [
        ("3\n1 4 10\n3\n2 15 20", "1 2"), ("2\n-5 0\n2\n-2 7", "-5 -2")]},
    15: {"title": "Переставить нули в конец", "cases": [
        ("7\n0 1 0 3 12 0 5", "1 3 12 5 0 0 0"), ("3\n0 0 0", "0 0 0"), ("3\n1 2 3", "1 2 3")]},
    16: {"title": "Минимальная длина окна с суммой не меньше S", "cases": [
        ("6\n2 3 1 2 4 3\n7", "2"), ("3\n1 1 1\n5", "0"), ("3\n1 4 4\n4", "1")]},
    17: {"title": "Самая длинная подстрока без повторов", "cases": [
        ("abcabcbb", "3"), ("bbbbb", "1"), ("", "0"), ("pwwkew", "3")]},
    18: {"title": "Сколько подотрезков имеют сумму не больше K", "cases": [
        ("5\n1 2 1 1 3\n4", "11"), ("3\n5 1 1\n3", "3"), ("4\n1 1 1 1\n10", "10")]},
    19: {"title": "Самая длинная серия после замены нулей", "cases": [
        ("7\n1 0 1 1 0 0 1\n2", "6"), ("4\n0 0 0 0\n1", "1"), ("5\n1 1 1 1 1\n0", "5")]},
    20: {"title": "Анаграммы", "cases": [
        ("listen\nsilent", "YES"), ("hello\nworld", "NO"), ("aabbcc\nabcabc", "YES")]},
    21: {"title": "Три числа с заданной суммой", "cases": [
        ("6\n-1 0 1 2 -1 -4\n0", "YES"), ("4\n1 2 3 4\n20", "NO"), ("3\n1 1 1\n3", "YES")]},
    22: {"title": "Площадки для дождевой воды", "cases": [
        ("12\n0 1 0 2 1 0 1 3 2 1 2 1", "6"), ("3\n3 0 3", "3"), ("4\n1 2 3 4", "0")]},
    23: {"title": "Минимальное окно со всеми буквами", "cases": [
        ("ADOBECODEBANC\nABC", "4"), ("a\naa", "0"), ("aa\naa", "2")]},
    24: {"title": "Количество инверсий", "cases": [
        ("5\n2 4 1 3 5", "3"), ("3\n3 2 1", "3"), ("4\n1 2 3 4", "0")]},
    25: {"title": "Максимальная сумма подотрезка", "cases": [
        ("9\n-2 1 -3 4 -1 2 1 -5 4", "6"), ("3\n-5 -2 -7", "-2"), ("1\n8", "8")]},
    26: {"title": "Первый пропавший положительный", "cases": [
        ("4\n1 2 0 4", "3"), ("3\n3 4 -1", "1"), ("5\n1 2 3 4 5", "6")]},
    27: {"title": "Максимальное произведение подотрезка", "cases": [
        ("4\n2 3 -2 4", "6"), ("3\n-2 0 -1", "0"), ("3\n-2 3 -4", "24")]},
    28: {"title": "Подотрезки ровно с K нечётными", "cases": [
        ("5\n1 1 2 1 1\n3", "2"), ("3\n2 4 6\n0", "6"), ("4\n2 1 2 1\n1", "6")]},
    29: {"title": "Медиана двух отсортированных списков", "cases": [
        ("2\n1 3\n1\n2", "2"), ("2\n1 2\n2\n3 4", "2.5"), ("0\n\n1\n7", "7")]},
    30: {"title": "Подсчёт подмассивов с K различными", "cases": [
        ("5\n1 2 1 2 3\n2", "7"), ("3\n1 2 3\n1", "3"), ("4\n1 1 1 1\n1", "10")]},
}


# Дополнительные случаи: края, повторы, нули, отрицательные значения и
# несколько одинаково хороших вариантов. Они дополняют основные примеры выше.
_EXTRA_CASES = {
    1: [("6\n-10 0 -3 8 8 1", "8")],
    2: [("6\n5 5 4 5 3 5", "0")],
    3: [("6\n1 2 3 4 5 6", "5")],
    4: [("6\n1 2 2 2 3 3", "3")],
    5: [("4\n0 1 0 -1", "0 1 1 0")],
    6: [("6\n1 2 3 4 5 6\n2 4", "12")],
    7: [("5\n2 3 -1 8 4", "3")],
    8: [("4\n1 1 5 10\n3\n1 2 10", "1 1 1 2 5 10 10")],
    9: [("5\n1 1 3 4 6\n2", "YES")],
    10: [("6\n-1 -1 0 0 2 2", "-1 0 2")],
    11: [("Не палиндром", "NO")],
    12: [("5\n2 2 2 2 2\n4", "2")],
    13: [("5\n1 -1 1 -1 1\n0", "6")],
    14: [("3\n1 5 10\n3\n2 6 9", "1 2")],
    15: [("5\n0 -1 0 -2 3", "-1 -2 3 0 0")],
    16: [("5\n1 2 3 4 5\n11", "3")],
    17: [("dvdf", "3")],
    18: [("3\n1 1 1\n2", "5")],
    19: [("6\n0 1 0 1 1 0\n1", "4")],
    20: [("abc\naabc", "NO")],
    21: [("5\n0 0 0 0 0\n0", "YES")],
    22: [("6\n4 2 0 3 2 5", "9")],
    23: [("aaflslflsldkalskaaa\naaa", "3")],
    24: [("4\n4 3 2 1", "6")],
    25: [("5\n1 2 3 -2 5", "9")],
    26: [("4\n7 8 9 11", "1")],
    27: [("4\n-1 -2 -3 0", "6")],
    28: [("5\n1 2 3 4 5\n2", "4")],
    29: [("1\n1\n2\n2 3", "2")],
    30: [("4\n1 2 3 4\n2", "3")],
}

for _number, _cases in _EXTRA_CASES.items():
    _TASKS[_number]["cases"].extend(_cases)


def _make_test(number):
    def test(func):
        _check(number, func)
    test.__name__ = f"test_task{number}"
    test.__doc__ = f"Проверить решение задачи {number}."
    return test


globals().update({f"test_task{i}": _make_test(i) for i in range(1, 31)})
