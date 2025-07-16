import unittest


from functions import (
    # импорт функций перестановок
    generate_random_permutation_fisher_yates,
    generate_random_permutation_floyd,
    generate_random_permutation_prefixcipher,
    generate_random_permutation_cyk,
    generate_random_permutation_Paloma,
    generate_random_permutation_PalomaOpt,

    # импорт функций выборок
    typical_sampling,
    generate_sampling_floyd_recursive,
    generate_sampling_floyd_iterative,
    generate_sampling_prefixcipher,
    generate_sampling_with_cyk,
    reservoir_sampling_list,
    generate_sampling_Paloma,
    generate_sampling_PalomaOpt,
    generate_sampling_hidden_shuffle,

    # функции векторов фиксированного веса
    fixed_weight,
    generate_fixed_weight_fisher_yates,
    generate_fixed_weight_prefixCipher,
    generate_fixed_weight_cyk,
    generate_fixed_vector_Paloma,

    # функция Feistel
    FeistelCipherOptimized
)

# тесты для перестановок
class TestPermutations(unittest.TestCase):
    def setUp(self):
        self.arr = [1, 2, 3, 4, 5]

    def test_fisher_yates(self):
        result = generate_random_permutation_fisher_yates(self.arr.copy())
        self.assertEqual(sorted(result), sorted(self.arr))
        self.assertNotEqual(result, self.arr)

    def test_floyd(self):
        result = generate_random_permutation_floyd(self.arr.copy())
        self.assertEqual(sorted(result), sorted(self.arr))
        self.assertNotEqual(result, self.arr)

    def test_prefixcipher(self):
        result = generate_random_permutation_prefixcipher(self.arr.copy())
        self.assertEqual(sorted(result), sorted(self.arr))
        self.assertNotEqual(result, self.arr)

    def test_cyk(self):
        result = generate_random_permutation_cyk(self.arr.copy())
        self.assertEqual(sorted(result), sorted(self.arr))
        self.assertNotEqual(result, self.arr)

    def test_paloma(self):
        result = generate_random_permutation_Paloma(self.arr.copy())
        self.assertEqual(sorted(result), sorted(self.arr))
        self.assertNotEqual(result, self.arr)

    def test_paloma_opt(self):
        result = generate_random_permutation_PalomaOpt(self.arr.copy())
        self.assertEqual(sorted(result), sorted(self.arr))
        self.assertNotEqual(result, self.arr)


# тесты для выборок
class TestSampling(unittest.TestCase):
    def setUp(self):
        self.arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.k = 5

    def test_typical_sampling(self):
        result = typical_sampling(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_floyd_recursive(self):
        result = generate_sampling_floyd_recursive(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_floyd_iterative(self):
        result = generate_sampling_floyd_iterative(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_prefixcipher(self):
        result = generate_sampling_prefixcipher(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_cyk(self):
        result = generate_sampling_with_cyk(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_reservoir_sampling(self):
        result = reservoir_sampling_list(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_paloma(self):
        result = generate_sampling_Paloma(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_paloma_opt(self):
        result = generate_sampling_PalomaOpt(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))

    def test_hidden_shuffle(self):
        result = generate_sampling_hidden_shuffle(self.arr, self.k)
        self.assertEqual(len(result), self.k)
        self.assertTrue(all(x in self.arr for x in result))


# тесты для векторов фиксированного веса
class TestFixedWeightVectors(unittest.TestCase):
    def test_fixed_weight(self):
        result = fixed_weight(n=100, m=10, q=1024, t=5, sigma_1=16)
        self.assertEqual(sum(result), 5)
        self.assertEqual(len(result), 100)

    def test_fisher_yates(self):
        result = generate_fixed_weight_fisher_yates(n=10, t=3)
        self.assertEqual(sum(result), 3)
        self.assertEqual(len(result), 10)

    def test_prefixcipher(self):
        result = generate_fixed_weight_prefixCipher(n=16, t=3)
        self.assertEqual(sum(result), 3)
        self.assertEqual(len(result), 16)

    def test_cyk(self):
        result = generate_fixed_weight_cyk(n=10, t=3)
        self.assertEqual(sum(result), 3)
        self.assertEqual(len(result), 10)

    def test_paloma(self):
        result = generate_fixed_vector_Paloma(t=3, n=10)
        self.assertEqual(sum(result), 3)
        self.assertEqual(len(result), 10)


# тесты для FeistelCipherOptimized
class TestFeistelCipher(unittest.TestCase):
    def test_get_permutation(self):
        cipher = FeistelCipherOptimized(k=10, r=3)
        perm = list(cipher.get_permutation())
        self.assertEqual(sorted(perm), list(range(10)))

    def test_get_sample(self):
        cipher = FeistelCipherOptimized(k=10, r=3)
        sample = list(cipher.get_sample(3))
        self.assertEqual(len(sample), 3)
        self.assertTrue(all(0 <= x < 10 for x in sample))

    def test_get_fixed_weight_vector(self):
        cipher = FeistelCipherOptimized(k=10, r=3)
        vec = cipher.get_fixed_weight_vector(t=3)
        self.assertEqual(sum(vec), 3)
        self.assertEqual(len(vec), 10)


# сам запуск
if __name__ == "__main__":
    unittest.main()