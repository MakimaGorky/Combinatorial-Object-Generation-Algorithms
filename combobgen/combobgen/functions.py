import random
import secrets
import math
from collections import defaultdict, Counter
from typing import List, Any, Iterator, Callable
import time
import os
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

from math import log , exp , floor
from random import uniform

import cryptography
from hpc import generate_hpc_functions
  
# region Перестановки

def fisher_yates_shuffle(arr):#вспомогательная функция
    """
    Алгоритм Fisher-Yates для случайной перестановки массива.
    Модифицирует исходный массив на месте.

    Args:
        arr: массив
        random_fun: функция генерации случайных данных
    """
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = secrets.randbelow(i+1)
        arr[i], arr[j] = arr[j], arr[i]

def generate_random_permutation_fisher_yates(arr): #алгоритм перестановки элементов в массиве Фишера Йетса
    fisher_yates_shuffle(arr)
    return arr

def floyd_permutation(n): #вспомогательная функция
    S = []
    for J in range(1, n + 1):
        T = random.randint(1, J)
        if T not in S:
            S.insert(0, T)  
        else:
            idx = S.index(T)
            S.insert(idx + 1, J)
    return S

def generate_random_permutation_floyd(arr): # алгоритм перестановки элементов в массиве Флойда
    perm = floyd_permutation(len(arr))
    return [arr[i-1] for i in perm]
                                            
def get_prefixcipher_permutation(k: int): # вспомогательная функция
    """
        Перестановки для k объектов на основе prefixCipher
        Args:
            длина перестановки k
        Returns: 
            Возвращает перестановку
    """
    blocksize = k.bit_length()
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
    for i in range(k):
        encrypted_val = encrypt(i, spice)
        encrypted_pairs.append((i, encrypted_val))
        print(f"E({i}) = {encrypted_val}")
    sorted_by_encrypted = sorted(encrypted_pairs, key=lambda item: item[1])
    permutation = [item[0] for item in sorted_by_encrypted]

    return permutation

def generate_random_permutation_prefixcipher(arr):  # алгоритм перестановки элементов в массиве Prefixcipher
    perm = get_prefixcipher_permutation(len(arr))
    return [arr[i] for i in perm]

def get_permutation_with_cyk(k: int) -> list[int]: # вспомогательная функция
    """
        Перестановки для [0..k-1] на основе CycleWalkCipher. В качестве блочного шифра используется hpc
    """
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

def generate_random_permutation_cyk(arr): # алгоритм перестановки элементов в массиве Cyk
    perm = get_permutation_with_cyk(len(arr))
    return [arr[i] for i in perm]

def getPermutation_Paloma(t: int) -> list[int]: # вспомогательная функция
    """
     Генерация перестановки в Paloma KEM
     """
    seed = os.urandom(32)   
    r_hat = []

    for i in range(16):
        sum = 0
        for j in range(16):
            cur = 16*i + 15 - j
            sum += ((seed[cur // 8] >> (cur % 8)) & 1)*(2**j)
        r_hat.append(sum)

    w = 0
    A = list(range(t))
    for i in range(t - 1, 0, -1):
        j = (r_hat[w]) % (i + 1)
        A[i], A[j] = A[j], A[i]
        w = (w + 1) % 16

    return A

def generate_random_permutation_Paloma(arr): # алгоритм перестановки элементов в массиве Paloma
    perm = getPermutation_Paloma(len(arr))
    return [arr[i] for i in perm]

def getPermutation_PalomaOpt(t: int) -> list[int]: #вспомогательная функция
    if t <= 0:
        return []
    A = list(range(t))
    r_bytes = os.urandom(32)    #SEED
    r_hat = []
    for w_val in range(16):
        r_hat_w_val = (r_bytes[2 * w_val + 1] << 8) | r_bytes[2 * w_val]
        r_hat.append(r_hat_w_val)

    w = 0
    for i in range(t - 1, 0, -1):
        j = (r_hat[w]) % (i + 1)
        A[i], A[j] = A[j], A[i]
        w = (w + 1) % 16

    return A

def generate_random_permutation_PalomaOpt(arr): # алгоритм перестановки элементов в массиве PalomaOpt
    perm = getPermutation_PalomaOpt(len(arr))
    return [arr[i] for i in perm]

# endregion


# region  Выборки

def typical_sampling(arr, k): # алгоритм выборки элементов в массиве (самый простой)
    """
    Args:
        arr: массив, из которого производится выборка
        k: количество элментов выборки (0<k<=len(arr))
    P.S в массиве должно быть достаточное количество различных k элементов, чтоб алгоритм не умер и не зациклился
    """
    s = []
    size = 0
    while size < k:
        t = secrets.randbelow(len(arr))
        if arr[t] not in s:
            s.append(arr[t])
            size+=1
    return s

def floyd_recursive(n, k): # вспомогательная функция
    """
    Args:
        n: количество элементов, из которого производится выборка
        k: количество элментов выборки (0 < k <= len(arr))
    P.S в массиве должно быть достаточное количество различных k элементов, чтоб алгоритм не умер и не зациклился
    """
    if k == 0:
        return []
    else:
        s = floyd_recursive(n-1, k-1)
        t = secrets.randbelow(n+1)
        if t not in s:
            s.append(t)
        else:
            s.append(n)
        return s

def generate_sampling_floyd_recursive(arr, k): # алгоритм выборки элементов в массиве (рекурсивный алгоритм Флойда)
    perm = floyd_recursive(len(arr), k)
    return [arr[i] for i in perm]
    
def floyd_iterative(n, k): #вспомогательная функция
    """
    Args:
        n: количество элементов, из которого производится выборка
        k: количество элментов выборки (0 < k <= len(arr))
    P.S в массиве должно быть достаточное количество различных k элементов, чтоб алгоритм не умер и не зациклился
    """
    s = []
    for j in range(n-k+1, n+1):
        t = secrets.randbelow(j)
        if t not in s:
            s.append(t)
        else:
            s.append(j)
    return s

def generate_sampling_floyd_iterative(arr, k): # алгоритм выборки элементов в массиве (итеративный алгоритм Флойда)
    perm = floyd_iterative(len(arr), k)
    return [arr[i] for i in perm]

def get_prefixcipher_sample(n: int, k: int): #вспоомогательная функция
    perm = get_prefixcipher_permutation(n)
    return perm[:k]

def generate_sampling_prefixcipher(arr, k): # алгоритм выборки элементов в массиве (prefixcipher)
    perm = get_prefixcipher_sample(len(arr), k)
    print(k)
    return [arr[i] for i in perm]
  
def get_sample_with_cyk(k: int, n: int) -> list[int]: #вспомогательная функция

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

    if n <= 0 or n > k:
        return []
    result_list = [0] * n

    for i in range(n):
        mapped_value = Cy_K(i)
        result_list[i] = mapped_value
    return result_list

def generate_sampling_with_cyk(arr, k): # алгоритм выборки элементов в массиве (cyk)
    perm = get_sample_with_cyk(len(arr), k)
    return [arr[i] for i in perm]


def reservoir_sampling_list(population: List[Any], k: int) -> List[Any]:  # алгоритм выборки элементов в массиве (резервуар)
    """
    Reservoir sampling для списка - выбирает k (0 < k <= n) случайных элементов.
    
    Args:
        population: исходный список элементов
        k: количество элементов для выборки
    
    Returns:
        список из k случайно выбранных элементов
    """
    
    # Инициализация резервуара первыми k элементами
    reservoir = population[:k]
    
    # Обработка остальных элементов
    for i in range(k, len(population)):
        # Генерируем случайный индекс от 0 до i включительно
        j = secrets.randbelow(i+1)
        
        # Если j < k, заменяем элемент в резервуаре
        if j < k:
            reservoir[j] = population[i]
    
    return reservoir


def getSamplePaloma(t: int, n: int): # вспомогательная функция
    """
     Генерация выборки в Paloma KEM.
     """
    perm = getPermutation_Paloma(t)
    return perm[:n]


def generate_sampling_Paloma(arr, k): # алгоритм выборки элементов в массиве (Palom)
    perm = getSamplePaloma(len(arr), k)
    return [arr[i] for i in perm]


def getSamplePalomaOpt(t: int, n: int): #SEED внутри permutation, вспомогательная функция
    perm = getPermutation_PalomaOpt(t)
    return perm[:n]

def generate_sampling_PalomaOpt(arr, k): # алгоритм выборки элементов в массиве (PalomOpt)
    perm = getSamplePalomaOpt(len(arr), k)
    return [arr[i] for i in perm]

def get_sample_hidden_shuffle(N, n):  # WOR from 0..N-1    вспоомгталеьная функция
    # STEP 1: compute H
    H = 0
    i = 0
    if N > n:
        H = n
    while i < n:
        q = 1.0 - float(N - n) / (N - i)
        i = i + int(log(uniform(0, 1), 1 - q))
        p_i = 1.0 - float(N - n) / (N - i)
        if i < n and uniform(0, 1) < p_i / q:
            H = H - 1
        i = i + 1

    L = n - H
    a = 1.0

    # STEP 2: draw high-items
    while H > 0:
        S_old = n + int(a * (N - n))
        a = a * uniform(0, 1) ** (1.0 / H)
        S = n + int(a * (N - n))
        if S < S_old:
            yield (N - 1) - S

        else:
            L = L + 1  # duplicate detected
        H = H - 1

    # STEP 3: draw low-items
    while L > 0:
        u = uniform(0, 1)
        s = 0
        F = float(L) / n
        while F < u and s < (n - L):
            F = 1 - (1 - float(L) / (n - s)) * (1 - F)
            s = s + 1
        L = L - 1
        n = n - s - 1
        yield (N - 1) - n

def generate_sampling_hidden_shuffle(arr, k): # алгоритм выборки элементов в массиве (hidden shuffle)
    perm = get_sample_hidden_shuffle(len(arr), k)
    return [arr[i] for i in perm]

# endregion

class FeistelCipherOptimized:


    """
    Класс, реализующий алгоритм генерации перестановок (и выборок) на основе сети фейстеля.

    В данной реализации в качестве Fj используется DES на ключе Kj.
    SEED = MASTER_KEY для DES.
    """

    def __init__(self, k: int, r: int):
        self.k = k
        self.r = r
        self.a = math.ceil(math.sqrt(self.k))
        self.b = math.ceil(math.sqrt(self.k))
        self.des_keys = [os.urandom(8) for _ in range(self.r)]  #MASTERKEY = SEED
        self.des_ciphers = [DES.new(key, DES.MODE_ECB) for key in self.des_keys]

    def _Fj(self, j: int, R: int) -> int:
        cipher = self.des_ciphers[j - 1]
        R_bytes = (R % (2 ** (8 * 8))).to_bytes(8, 'big')
        encrypted_R_bytes = cipher.encrypt(R_bytes)
        return int.from_bytes(encrypted_R_bytes, 'big')

    def fe(self, m: int) -> int:
        L = m % self.a
        R = m // self.a

        for j in range(1, self.r + 1):
            if j % 2 != 0:
                t = (L + self._Fj(j, R)) % self.a
            else:
                t = (L + self._Fj(j, R)) % self.b
            L_next = R
            R_next = t
            L = L_next
            R = R_next

        if self.r % 2 != 0:
            result = self.a * L + R
        else:
            result = self.a * R + L
        return result

    def FE(self, m: int) -> int:
        t = self.fe(m)
        if t < self.k:
            return t
        else:
            return self.FE(t)

    def get_permutation(self):
        for i in range(self.k):
            yield self.FE(i)

    def get_sample(self, t):
        for i in range(t):
            yield self.FE(i)
    def get_fixed_weight_vector(self, t):
        vector = (self.k-t)*[0]+t*[1]
        perm = list(self.get_permutation())
        result = [vector[i] for i in perm]
        return result



                                   # reservoir

# region Векторы фиксированного веса 

#  Вспомогательная функция 
def has_duplicates(arr):
    """
    Проверка на наличие дубликатов в массив.

    Args: 
        arr: массив
    """
    seen = {}
    for item in arr:
        if item in seen:
            return True
        else:
            seen[item] = 1
    return False

def fixed_weight(n, m, q ,t, sigma_1):
    """
    Алгоритм генерации вектора фиксированнного веса (из classic McEliece)

    Args:
        n: длина возвращаемого вектора; n<=q
        m: log_2(q)
        q: размерность пространства; q=2^m
        t: вес возвращаемого вектора; t>=2; mt<n
        sigma_1: парамер криптосистемы; sigma_1>=m

    Returns:
        вектор длины n веса t

    """
    tau = t
    if n == q:
        tau = t
    elif q/2 <= n and n < q:
        tau = 2*t
    b = []
    for i in range(sigma_1*tau):
        b.append(secrets.randbelow(2))
    d = []
    for j in range(tau):
        d_j = 0
        for i in range(m):
            d_j += b[sigma_1*j+i]*(2**i)
        d.append(d_j)
    a = []  
    for i in range(len(d)):
        if d[i]<n:
            a.append(d[i])
        if(len(a) == t):
            break
    if(len(a) < t):
        print("всё по новой")
        return fixed_weight(n, m, q ,t, sigma_1)

    if (has_duplicates(a)):
        print("всё по новой")
        return fixed_weight(n, m, q ,t, sigma_1)
    
    e = [0] * n
    for index in a:
        e[index] = 1
    return e

def generate_fixed_weight_fisher_yates(n, t):
    """
    Алгоритм генерации вектора фиксированнного веса (из fisher_yates_shuffle)

    Args:
        n: длина вектора
        t: вес вектора
        random_fun: функция генерации случайных битов
    Returns: 
        вектор длины n веса t
    """
    vector = (n-t)*[0]+t*[1]
    fisher_yates_shuffle(vector)
    return vector

def generate_fixed_weight_prefixCipher(n, t):
    """
    ????Алгоритм генерации вектора фиксированнного веса (из get_prefixcipher_permutation)

    Args:
        ? n: длина вектора (длина должна быть степенью 2)
        t: вес вектора
    Returns: 
        вектор длины n веса t
    """
    vector = (n-t)*[0]+t*[1]
    perm = get_prefixcipher_permutation(n)
    result = [vector[i] for i in perm]
    return result

def generate_fixed_weight_cyk(n, t):
    """
    Алгоритм генерации вектора фиксированнного веса (из get_permutation_with_cyk)

    Args:
        n: длина вектора
        t: вес вектора
    Returns: 
        вектор длины n веса t
    """
    vector = (n-t)*[0]+t*[1]
    perm = get_permutation_with_cyk(n)
    result = [vector[i] for i in perm]
    return result

def generate_fixed_vector_Paloma(t: int, n:int): #SEED внутри permutation
    """
     Генерация вектора веса t длины n в Paloma KEM.
     """
    vect = [0] * n
    perm = getPermutation_PalomaOpt(n)
    for i in range(t):
        curpermitem = perm[i]
        vect[curpermitem] = 1
    return vect

# endregion


print(generate_sampling_hidden_shuffle([3,45,3,2,4,7,8,9], 3))