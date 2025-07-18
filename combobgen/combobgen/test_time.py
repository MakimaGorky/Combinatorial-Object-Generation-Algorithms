import time
import random
import timeit  # Импортируем модуль timeit
from collections import defaultdict
import math
import sys

#sys.setrecursionlimit(5000)

# Убедитесь, что functions.py находится в той же директории
from functions import (
    generate_random_permutation_fisher_yates,
    floyd_permutation,
    get_prefixcipher_permutation,
    get_permutation_with_cyk,
    generate_permutation_prefixcipherAES,
    getPermutation_Paloma,
    getPermutation_PalomaOpt,

    typical_sampling,
    #floyd_recursive,
    floyd_iterative,
    get_prefixcipher_sample,
    get_sample_with_cyk,
    reservoir_sampling_list,
    getSamplePaloma,
    getSamplePalomaOpt,
    get_sample_hidden_shuffle,

    fixed_weight,
    generate_fixed_weight_fisher_yates,
    generate_fixed_weight_prefixCipher,
    generate_fixed_weight_cyk,
    generate_fixed_vector_Paloma,
    FeistelCipherOptimized
)

def run_and_measure_avg(stmt, setup, number=100, repeat=5):
    """
    Запускает код stmt 'number' раз, повторяет 'repeat' раз
    и возвращает среднее время выполнения одного прогона (минимальное из repeat-ов).
    """
    times = timeit.repeat(stmt, setup, number=number, repeat=repeat)
    # Возвращаем минимальное время из повторений, деленное на количество прогонов,
    # чтобы получить среднее время одного вызова функции.
    return min(times) / number

# Диапазоны N для тестирования
N_VALUES = [10**2, 10**3, 10**4, 10**5] 

# Базовое количество прогонов и повторений 
NUMBER_RUNS = 100 
REPEAT_TIMES = 1


# Тесты для функций перестановок
def test_permutations():
    print("--- Тестирование алгоритмов перестановок ---")
    
    for N in N_VALUES:
        print(f"\n--- Тесты для N = {N}, NUMBER_RUNS = {NUMBER_RUNS} ---")

        tests = defaultdict(dict)

        # Fisher-Yates
        setup_code = f"from functions import generate_random_permutation_fisher_yates; arr_copy = list(range({N}))"
        stmt_code = "generate_random_permutation_fisher_yates(arr_copy)"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Fisher-Yates Shuffle"]["time"] = avg_time
        tests["Fisher-Yates Shuffle"]["N"] = N

        # Floyd Permutation
        setup_code = f"from functions import floyd_permutation; arr_copy = list(range({N}))"
        stmt_code = f"floyd_permutation({N})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Floyd Permutation"]["time"] = avg_time
        tests["Floyd Permutation"]["N"] = N

        # Prefixcipher Permutation
       
        setup_code = f"from functions import get_prefixcipher_permutation; arr_prefixcipher = list(range({N}))"
        stmt_code = f"get_prefixcipher_permutation({N})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Prefixcipher Permutation"]["time"] = avg_time
        tests["Prefixcipher Permutation"]["N"] = N
    

        # Cyk Permutation
        
        setup_code = f"from functions import get_permutation_with_cyk; arr_cyk = list(range({N}))"
        stmt_code = f"get_permutation_with_cyk({N})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Cyk Permutation"]["time"] = avg_time
        tests["Cyk Permutation"]["N"] = N

        # AES Permutation
        setup_code = f"from functions import generate_permutation_prefixcipherAES; arr_copy = list(range({N}))"
        stmt_code = f"generate_permutation_prefixcipherAES({N})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["AES Permutation"]["time"] = avg_time
        tests["AES Permutation"]["N"] = N
        
        
        # Paloma Permutation
        setup_code = f"from functions import getPermutation_Paloma; arr_copy = list(range({N}))"
        stmt_code = f"getPermutation_Paloma({N})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Paloma Permutation"]["time"] = avg_time
        tests["Paloma Permutation"]["N"] = N

        # PalomaOpt Permutation
        setup_code = f"from functions import getPermutation_PalomaOpt; arr_copy = list(range({N}))"
        stmt_code = f"getPermutation_PalomaOpt({N})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["PalomaOpt Permutation"]["time"] = avg_time
        tests["PalomaOpt Permutation"]["N"] = N

        # Feistel Cipher Optimized Permutation
        # k для FeistelCipher должен быть равен N
        feistel_k = N
        feistel_r = 8
        setup_code = f"""
from functions import FeistelCipherOptimized
feistel_cipher = FeistelCipherOptimized(k={feistel_k}, r={feistel_r})
"""
        stmt_code = "list(feistel_cipher.get_permutation())"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Feistel Cipher Optimized Permutation"]["time"] = avg_time
        tests["Feistel Cipher Optimized Permutation"]["N"] = feistel_k

        for name, data in tests.items():
            if isinstance(data["time"], str):
                 print(f"{name} (N={data['N']}): {data['time']}")
            else:
                print(f"{name} (N={data['N']}): {data['time']:.6f} секунд (среднее за {NUMBER_RUNS} прогонов)")


# Тесты для функций выборки
def test_samplings():
    print("\n--- Тестирование алгоритмов выборки ---")

    for N in N_VALUES:
        K = max(1, int(0.1 * N)) # Количество элементов для выборки (10% от N)
        print(f"\n--- Тесты для N = {N}, K = {K}, NUMBER_RUNS = {NUMBER_RUNS} ---")

        tests = defaultdict(dict)

        # Typical Sampling
        setup_code = f"from functions import typical_sampling; arr_copy = list(range({N}))"
        stmt_code = f"typical_sampling(arr_copy, {K})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Typical Sampling"]["time"] = avg_time
        tests["Typical Sampling"]["N"] = N
        tests["Typical Sampling"]["K"] = K

    
        
        
        # Floyd Iterative Sampling
        setup_code = f"from functions import floyd_iterative; arr_copy = list(range({N}))"
        stmt_code = f"floyd_iterative({N}, {K})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Floyd Iterative Sampling"]["time"] = avg_time
        tests["Floyd Iterative Sampling"]["N"] = N
        tests["Floyd Iterative Sampling"]["K"] = K

        
        setup_code = f"from functions import get_prefixcipher_sample; arr_prefixcipher_sample = list(range({N}))"
        stmt_code = f"get_prefixcipher_sample({N}, {K})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Prefixcipher Sampling"]["time"] = avg_time
        tests["Prefixcipher Sampling"]["N"] = N
        tests["Prefixcipher Sampling"]["K"] = K
        
        
        # Cyk Sampling
        
        setup_code = f"from functions import get_sample_with_cyk; arr_cyk_sample = list(range({N}))"
        stmt_code = f"get_sample_with_cyk({N}, {K})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Cyk Sampling"]["time"] = avg_time
        tests["Cyk Sampling"]["N"] = N
        tests["Cyk Sampling"]["K"] = K
        

        # Reservoir Sampling
        setup_code = f"from functions import reservoir_sampling_list; arr_copy = list(range({N}))"
        stmt_code = f"reservoir_sampling_list(arr_copy, {K})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Reservoir Sampling"]["time"] = avg_time
        tests["Reservoir Sampling"]["N"] = N
        tests["Reservoir Sampling"]["K"] = K

        # Paloma Sampling
        setup_code = f"from functions import getSamplePaloma; arr_copy = list(range({N}))"
        stmt_code = f"getSamplePaloma({N}, {K})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Paloma Sampling"]["time"] = avg_time
        tests["Paloma Sampling"]["N"] = N
        tests["Paloma Sampling"]["K"] = K

        # PalomaOpt Sampling
        setup_code = f"from functions import getSamplePalomaOpt; arr_copy = list(range({N}))"
        stmt_code = f"getSamplePalomaOpt({N}, {K})"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["PalomaOpt Sampling"]["time"] = avg_time
        tests["PalomaOpt Sampling"]["N"] = N
        tests["PalomaOpt Sampling"]["K"] = K

        # Hidden Shuffle Sampling
        setup_code = f"from functions import get_sample_hidden_shuffle; arr_copy = list(range({N}))"
        stmt_code = f"list(get_sample_hidden_shuffle({N}, {K}))"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Hidden Shuffle Sampling"]["time"] = avg_time
        tests["Hidden Shuffle Sampling"]["N"] = N
        tests["Hidden Shuffle Sampling"]["K"] = K

        # Feistel Cipher Optimized Sampling
        feistel_k_sample = N
        feistel_t_sample = K
        setup_code = f"""
from functions import FeistelCipherOptimized
feistel_cipher_sample = FeistelCipherOptimized(k={feistel_k_sample}, r=8)
"""
        stmt_code = f"list(feistel_cipher_sample.get_sample({feistel_t_sample}))"
        avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
        tests["Feistel Cipher Optimized Sampling"]["time"] = avg_time
        tests["Feistel Cipher Optimized Sampling"]["N"] = feistel_k_sample
        tests["Feistel Cipher Optimized Sampling"]["K"] = feistel_t_sample

        for name, data in tests.items():
            if isinstance(data["time"], str):
                print(f"{name} (N={data['N']}, K={data['K']}): {data['time']}")
            else:
                print(f"{name} (N={data['N']}, K={data['K']}): {data['time']:.6f} секунд (среднее за {NUMBER_RUNS} прогонов)")


# Тесты для функций векторов фиксированного веса
def test_fixed_weight_vectors():
    print("\n--- Тестирование алгоритмов генерации векторов фиксированного веса ---")

    for N in N_VALUES:
        T = max(1, int(0.1 * N)) # Вес вектора (10% от N)
        print(f"\n--- Тесты для N = {N}, T = {T}, NUMBER_RUNS = {NUMBER_RUNS} ---")

        tests = defaultdict(dict)

        # Fixed Weight (Classic McEliece)
        # Параметры McEliece должны быть тщательно подобраны.
        # Для простоты, n_mc = N, t_mc = T
        n_mc = N
        t_mc = T
        m_mc = 9
        q_mc = 2**m_mc
        sigma_1_mc = m_mc + 30 # sigma_1 >= m

        # Проверка условий McEliece
        # t>=2, mt<n, n<=q, sigma_1>=m
        if t_mc < 2 or m_mc * t_mc >= n_mc or n_mc > q_mc or sigma_1_mc < m_mc:
            tests["Fixed Weight (Classic McEliece)"]["time"] = "Skipped (McEliece constraints not met for N,T)"
            tests["Fixed Weight (Classic McEliece)"]["N"] = N
            tests["Fixed Weight (Classic McEliece)"]["T"] = T
        else:
            setup_code = f"from functions import fixed_weight"
            stmt_code = f"fixed_weight({n_mc}, {m_mc}, {q_mc}, {t_mc}, {sigma_1_mc})"
            avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
            tests["Fixed Weight (Classic McEliece)"]["time"] = avg_time
            tests["Fixed Weight (Classic McEliece)"]["N"] = N
            tests["Fixed Weight (Classic McEliece)"]["T"] = T


#         # Fixed Weight Fisher-Yates
#         setup_code = f"from functions import generate_fixed_weight_fisher_yates"
#         stmt_code = f"generate_fixed_weight_fisher_yates({N}, {T})"
#         avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
#         tests["Fixed Weight Fisher-Yates"]["time"] = avg_time
#         tests["Fixed Weight Fisher-Yates"]["N"] = N
#         tests["Fixed Weight Fisher-Yates"]["T"] = T

#         # Fixed Weight PrefixCipher
        
#         setup_code = f"from functions import generate_fixed_weight_prefixCipher"
#         stmt_code = f"generate_fixed_weight_prefixCipher({N}, {T})"
#         avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
#         tests["Fixed Weight PrefixCipher"]["time"] = avg_time
#         tests["Fixed Weight PrefixCipher"]["N"] = N
#         tests["Fixed Weight PrefixCipher"]["T"] = T
        

#         # Fixed Weight Cyk
#         setup_code = f"from functions import generate_fixed_weight_cyk"
#         stmt_code = f"generate_fixed_weight_cyk({N}, {T})"
#         avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
#         tests["Fixed Weight Cyk"]["time"] = avg_time
#         tests["Fixed Weight Cyk"]["N"] = N
#         tests["Fixed Weight Cyk"]["T"] = T
        

#         # Fixed Weight Paloma
#         setup_code = f"from functions import generate_fixed_vector_Paloma"
#         stmt_code = f"generate_fixed_vector_Paloma({T}, {N})"
#         avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
#         tests["Fixed Weight Paloma"]["time"] = avg_time
#         tests["Fixed Weight Paloma"]["N"] = N
#         tests["Fixed Weight Paloma"]["T"] = T

#         # Feistel Cipher Optimized Fixed Weight Vector
#         feistel_k_fw = N
#         feistel_t_fw = T
#         setup_code = f"""
# from functions import FeistelCipherOptimized
# feistel_cipher_fw = FeistelCipherOptimized(k={feistel_k_fw}, r=8)
# """
#         stmt_code = f"feistel_cipher_fw.get_fixed_weight_vector({feistel_t_fw})"
#         avg_time = run_and_measure_avg(stmt_code, setup_code, NUMBER_RUNS, REPEAT_TIMES)
#         tests["Feistel Cipher Optimized Fixed Weight Vector"]["time"] = avg_time
#         tests["Feistel Cipher Optimized Fixed Weight Vector"]["N"] = feistel_k_fw
#         tests["Feistel Cipher Optimized Fixed Weight Vector"]["T"] = feistel_t_fw

        for name, data in tests.items():
            if isinstance(data["time"], str):
                print(f"{name} (N={data['N']}, T={data['T']}): {data['time']}")
            else:
                print(f"{name} (N={data['N']}, T={data['T']}): {data['time']:.6f} секунд (среднее за {NUMBER_RUNS} прогонов)")


if __name__ == "__main__":
    #test_permutations()
    #test_samplings()
    test_fixed_weight_vectors()