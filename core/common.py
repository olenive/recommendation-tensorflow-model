import numpy as np

def top10(arr, num_top=10):
    idx = (-arr).argsort()[:num_top]
    book_numbers = np.array(range(1, 1001))
    return book_numbers[idx]
