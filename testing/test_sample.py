def multiply(a, b):
    return a * b

def test_multiply_positive():
    assert multiply(2, 3) == 6

def test_multiply_zero():
    assert multiply(5, 0) == 0

def test_faulty():
    assert multiply(5, 1) == 0