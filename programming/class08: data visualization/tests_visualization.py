"""Автотесты к уроку «Визуализация данных: matplotlib и seaborn».

Пример запуска в ноутбуке:

    from tests_visualization import test_task1
    test_task1(make_age_hist)

Функции получают DataFrame и должны вернуть созданный объект: обычно Axes,
а в задаче 10 — Figure. Тесты не открывают окна с графиками.
"""

import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

__all__ = [f"test_task{i}" for i in range(1, 26)]


def _df():
    return pd.DataFrame({
        "survived": [0, 1, 1, 0, 1, 0, 1, 0],
        "pclass": [3, 1, 2, 3, 1, 2, 3, 1],
        "sex": ["male", "female", "female", "male", "female", "male", "female", "male"],
        "age": [22., 38., 26., 35., 18., 40., 8., 50.],
        "fare": [7.25, 71.28, 13., 8.05, 53.1, 20., 21., 60.],
        "embarked": ["S", "C", "S", "S", "C", "Q", "S", "C"],
        "sibsp": [1, 1, 0, 0, 0, 0, 3, 1],
    })


def _call(func, *args):
    data = args[0]
    before = data.copy(deep=True) if isinstance(data, pd.DataFrame) else None
    with contextlib.redirect_stdout(io.StringIO()):
        result = func(*args)
    if before is not None and not data.equals(before):
        raise AssertionError("функция изменила исходную таблицу; работайте с copy() при необходимости")
    return result


def _run(number, name, func, check):
    if not callable(func):
        print(f"❌ test_task{number}: передайте функцию без скобок")
        return
    plt.close("all")
    try:
        check(func)
    except Exception as exc:
        print(f"Задача {number}. {name}: ❌ {type(exc).__name__}: {exc}")
    else:
        print(f"Задача {number}. {name}: все проверки пройдены ✅")
    finally:
        plt.close("all")


def _axes(value):
    if not isinstance(value, Axes):
        raise AssertionError("верните объект Axes (например, переменную ax), а не вызывайте только plt.show()")
    return value


def test_task1(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert len(ax.patches) > 0, "гистограмма должна содержать столбики"
        assert ax.get_xlabel(), "подпишите ось X через set_xlabel или set"
        assert ax.get_ylabel(), "подпишите ось Y через set_ylabel или set"
    _run(1, "make_age_hist", func, check)


def test_task2(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        heights = sorted(round(p.get_height(), 6) for p in ax.patches)
        assert heights == [0.0, 1.0], "столбцы должны показывать доли выживших по полу"
        low, high = ax.get_ylim()
        assert low <= 0 and high >= 1, "для долей установите шкалу Y от 0 до 1"
    _run(2, "make_survival_bar", func, check)


def test_task3(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "scatter-график должен содержать коллекцию точек"
        assert ax.get_xlabel() and ax.get_ylabel(), "подпишите обе оси"
    _run(3, "make_age_fare_scatter", func, check)


def test_task4(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.get_title(), "добавьте понятный заголовок"
        assert len(ax.get_xticklabels()) >= 3, "на оси X должны быть три класса билета"
    _run(4, "make_fare_boxplot", func, check)


def test_task5(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert len(ax.patches) >= 3, "seaborn barplot должен нарисовать столбики для классов"
        assert ax.get_ylabel(), "подпишите ось доли выживших"
    _run(5, "seaborn_class_bar", func, check)


def test_task6(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "seaborn scatterplot должен содержать точки"
        assert ax.get_legend() is not None, "используйте hue='survived', чтобы появилась легенда"
    _run(6, "seaborn_survival_scatter", func, check)


def test_task7(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.patches or ax.lines, "histplot/kdeplot должен что-то нарисовать"
        assert ax.get_legend() is not None, "разделите классы через hue='pclass'"
    _run(7, "seaborn_age_hist", func, check)


def test_task8(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.lines, "line-график должен содержать линию"
        y = ax.lines[0].get_ydata()
        assert np.all((y >= 0) & (y <= 1)), "по Y должны быть доли выживших"
        assert ax.get_ylim()[0] <= 0 and ax.get_ylim()[1] >= 1, "для долей установите Y от 0 до 1"
    _run(8, "plot_age_group_rate", func, check)


def test_task9(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "heatmap создаёт цветную матрицу (collection)"
        assert ax.get_xlabel() and ax.get_ylabel(), "подпишите, какие группы на осях"
    _run(9, "seaborn_survival_heatmap", func, check)


def test_task10(func):
    def check(fn):
        fig = _call(fn, _df())
        if not isinstance(fig, Figure):
            raise AssertionError("верните Figure: это объект fig из plt.subplots(2, 2, ...)")
        assert len(fig.axes) >= 4, "на дашборде должны быть четыре графика"
        assert all(ax.get_title() for ax in fig.axes[:4]), "у каждого графика должен быть заголовок"
    _run(10, "make_dashboard", func, check)


def test_task11(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert len(ax.patches) >= 3, "countplot должен показать порты посадки"
        assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel(), "добавьте заголовок и подписи"
    _run(11, "count_by_embarked", func, check)


def test_task12(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "нужен scatter-график"
        assert ax.get_yscale() == "log", "используйте логарифмическую шкалу цены: ax.set_yscale('log')"
    _run(12, "fare_log_scatter", func, check)


def test_task13(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.get_title(), "добавьте заголовок"
        assert len(ax.get_xticklabels()) >= 3, "сравните три класса билета по оси X"
    _run(13, "seaborn_age_boxplot", func, check)


def test_task14(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert len(ax.patches) >= 6, "нужны столбики для класса и пола (hue='sex')"
        assert ax.get_legend() is not None, "hue должен создать легенду"
    _run(14, "seaborn_survival_grouped_bar", func, check)


def test_task15(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "сначала нарисуйте точки"
        assert ax.texts, "подпишите самый дорогой билет через ax.annotate или ax.text"
    _run(15, "annotate_max_fare", func, check)


def test_task16(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert len(ax.lines) >= 2, "нарисуйте отдельную линию для каждого пола"
        assert ax.get_legend() is not None, "у линий должна быть легенда"
    _run(16, "survival_lines_by_sex", func, check)


def test_task17(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.lines, "kdeplot должен нарисовать кривую плотности"
        assert ax.get_legend() is not None, "разделите кривые через hue='sex'"
    _run(17, "seaborn_age_kde", func, check)


def test_task18(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "heatmap должна содержать цветную матрицу"
        assert ax.texts, "добавьте числа в ячейки: annot=True"
    _run(18, "embarked_class_heatmap", func, check)


def test_task19(func):
    def check(fn):
        value = _call(fn, _df())
        if not hasattr(value, "axes") or len(np.ravel(value.axes)) < 2:
            raise AssertionError("верните объект FacetGrid минимум с двумя отдельными графиками")
    _run(19, "facet_age_by_class", func, check)


def test_task20(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "нужна тепловая карта корреляций"
        assert ax.get_title(), "добавьте заголовок"
    _run(20, "correlation_heatmap", func, check)


def test_task21(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert len(ax.patches) == 2, "круговая диаграмма должна иметь два сектора — выжил/не выжил"
        assert ax.get_title(), "добавьте заголовок"
    _run(21, "survival_pie", func, check)


def test_task22(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.lines, "после группировки нужна линия"
        assert ax.get_ylim()[0] <= 0 and ax.get_ylim()[1] >= 1, "по Y должны быть доли от 0 до 1"
    _run(22, "survival_by_sibsp", func, check)


def test_task23(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert ax.collections, "seaborn violinplot создаёт коллекции с формой распределения"
        assert ax.get_title(), "добавьте заголовок"
    _run(23, "seaborn_fare_violin", func, check)


def test_task24(func):
    def check(fn):
        ax = _axes(_call(fn, _df()))
        assert len(ax.patches) >= 3, "нарисуйте категории класса билета"
        assert ax.get_title(), "добавьте заголовок"
    _run(24, "styled_class_count", func, check)


def test_task25(func):
    def check(fn):
        fig = _call(fn, _df())
        if not isinstance(fig, Figure):
            raise AssertionError("верните объект Figure")
        assert len(fig.axes) >= 6, "сделайте дашборд минимум из шести графиков (сетка 2×3)"
        assert all(ax.get_title() for ax in fig.axes[:6]), "добавьте заголовок каждому графику"
    _run(25, "make_final_dashboard", func, check)
