"""python -m unittest sample.Testing
Verbose: $python -m unittest -v sample.Testing"""

import unittest


def addition(x, y):
    return x + y


class Testing(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(addition(2, 2), 4)

    def test_lstrip(self):  # testing for left stripping
        self.assertEqual('   hello '.lstrip(), 'hello ')

    def test_isupper(self):  # testing for isupper
        self.assertTrue('HELLO'.isupper())
        self.assertFalse('HELlO'.isupper())

    def test_error_here(self):
        self.assertTrue(False)

    def test_split(self):  # testing for split
        self.assertEqual('Hello World'.split(), ['Hello', 'World'])
        with self.assertRaises(TypeError):
            'Hello World'.split('@')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
