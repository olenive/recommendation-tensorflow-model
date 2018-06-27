import numpy as np
from numpy.testing import assert_array_equal

from core import common


def test_top10():
    book_scores = np.array([0.0001] * 1000)
    book_scores[10] = 0.9
    book_scores[9] = 0.88
    book_scores[100] = 0.85
    book_scores[200] = 0.84
    book_scores[300] = 0.7
    book_scores[400] = 0.6
    book_scores[700] = 0.5
    book_scores[800] = 0.4
    book_scores[90] = 0.3
    book_scores[0] = 0.2
    book_scores[101] = 0.1
    book_scores[102] = 0.01
    book_scores[91] = 0.0
    expected_book_numbers = np.array([10, 9, 100, 200, 300, 400, 700, 800, 90, 0]) + 1
    result = common.top10(book_scores)
    assert_array_equal(result, expected_book_numbers)
