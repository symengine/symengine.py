from symengine.utilities import raises

from symengine import Integer

def test_integer():
    i = Integer(5)
    assert str(i) == "5"

def test_integer_long():
    i = Integer(123434444444444444444)
    assert str(i) == "123434444444444444444"

def test_integer_string():
    assert Integer("133") == 133

def test_smallfloat_valid():
    i = Integer(7.5)
    assert str(i) == "7"

def test_bigfloat_valid():
    i = Integer(13333333333333334.5)
    assert str(i) == "13333333333333334"
