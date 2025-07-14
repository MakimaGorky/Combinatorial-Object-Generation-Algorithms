import random
import math
from collections import defaultdict, Counter
from typing import List, Any, Iterator, Callable
import time

import cryptography

"""

TODO:
- перенести все функции и унифицировать их интерфейс для тестирования
- уву

"""

def fisher_yates_shuffle(arr):
    """
    Алгоритм Fisher-Yates для случайной перестановки массива.
    Модифицирует исходный массив на месте.
    """
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]


def generate_random_permutation(n):
    """Генерирует случайную перестановку чисел от 0 до n-1."""
    arr = list(range(n))
    fisher_yates_shuffle(arr)
    return arr


def reservoir_sampling_list(population: List[Any], k: int) -> List[Any]:
    """
    Reservoir sampling для списка - выбирает k случайных элементов.
    
    Args:
        population: исходный список элементов
        k: количество элементов для выборки
    
    Returns:
        список из k случайно выбранных элементов
    """
    if k >= len(population):
        result = population.copy()
        random.shuffle(result)
        return result
    
    # Инициализация резервуара первыми k элементами
    reservoir = population[:k]
    
    # Обработка остальных элементов
    for i in range(k, len(population)):
        # Генерируем случайный индекс от 0 до i включительно
        j = random.randint(0, i)
        
        # Если j < k, заменяем элемент в резервуаре
        if j < k:
            reservoir[j] = population[i]
    
    return reservoir


def reservoir_sampling_stream(stream: Iterator[Any], k: int) -> List[Any]:
    """
    Reservoir sampling для потока данных - работает с итераторами.
    
    Args:
        stream: итератор элементов
        k: количество элементов для выборки
    
    Returns:
        список из k случайно выбранных элементов
    """
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            # Заполняем резервуар первыми k элементами
            reservoir.append(item)
        else:
            # Для каждого нового элемента решаем, заменить ли им случайный элемент
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    
    return reservoir


def reservoir_sampling_weighted(population: List[Any], weights: List[float], k: int) -> List[Any]:
    """
    Weighted reservoir sampling - выборка с весами.
    
    Args:
        population: исходный список элементов
        weights: веса элементов
        k: количество элементов для выборки
    
    Returns:
        список из k случайно выбранных элементов с учетом весов
    """
    if len(population) != len(weights):
        raise ValueError("Длины population и weights должны совпадать")
    
    if k >= len(population):
        return population.copy()
    
    # Алгоритм A-ExpJ для взвешенной выборки
    reservoir = []
    keys = []
    
    for i, (item, weight) in enumerate(zip(population, weights)):
        if weight <= 0:
            continue
            
        # Генерируем ключ для элемента
        key = random.random() ** (1.0 / weight)
        
        if len(reservoir) < k:
            reservoir.append(item)
            keys.append(key)
        else:
            # Находим минимальный ключ
            min_idx = keys.index(min(keys))
            if key > keys[min_idx]:
                reservoir[min_idx] = item
                keys[min_idx] = key
    
    return reservoir