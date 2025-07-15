import random
import math
from collections import defaultdict, Counter
from typing import List, Any, Iterator, Callable
import time

import cryptography
from hpc import generate_hpc_functions


"""

TODO:
- перенести все функции и унифицировать их интерфейс для тестирования
- уву

"""

                                    # fisher-yates
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


                                    # reservoir
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



                                            # hpc
def get_prefixcipher_permutation(k: int): #перестановки для 2^N объектов на основе prefixCipher
    blocksize = k
    backup = 0
    key_length = 128
    # key = 0xf3aef8062681d980c14bd5915305f319
    spice = 0x4957df9f02329f2d07289bb61a440e059f9c5dcb93048b5686208a26403c5e7f706d0051cdb0d7bb8f0c6e4962e43023a0b02b363ffa0b53abf6d3f4f848f5e9

    # random keygen
    key_length_bytes = key_length // 8
    random_bytes = os.urandom(key_length_bytes)
    random_key_int = int.from_bytes(random_bytes, byteorder='big')
    encrypt, decrypt = generate_hpc_functions(random_key_int, blocksize, key_length, backup)
    encrypted_pairs = []
    for i in range(2**k):
        encrypted_val = encrypt(i, spice)
        encrypted_pairs.append((i, encrypted_val))
        print(f"E({i}) = {encrypted_val}")
    sorted_by_encrypted = sorted(encrypted_pairs, key=lambda item: item[1])
    permutation = [item[0] for item in sorted_by_encrypted]

    return permutation

def get_permutation_with_cyk(k: int) -> list[int]: #перестановки для [0..k-1] на основе CycleWalkCipher. В качестве блочного шифра используется hpc

    #creating parameters, cipher - don't include when measuring time. include randomizing spice.
    blocksize = k.bit_length()
    backup = 0
    key_length = 128
    key_length_bytes = key_length // 8
    random_bytes = os.urandom(key_length_bytes)
    random_key_int = int.from_bytes(random_bytes, byteorder='big')
    spice = 0x4957df9f02329f2d07289bb61a440e059f9c5dcb93048b5686208a26403c5e7f706d0051cdb0d7bb8f0c6e4962e43023a0b02b363ffa0b53abf6d3f4f848f5e9
    encrypt, decrypt = generate_hpc_functions(random_key_int, blocksize, key_length, backup)

    def Cy_K(m: int) -> int:
        t = encrypt(m, spice)
        if t < k:
            return t
        else:
            return Cy_K(t)

    if k <= 0:
        return []
    result_list = [0] * k
    for i in range(k):
        mapped_value = Cy_K(i)
        result_list[i] = mapped_value
    return result_list

def is_permutation_of_range(input_list: list) -> bool:
    expected_set = set(range(len(input_list)))
    actual_set = set(input_list)
    return actual_set == expected_set