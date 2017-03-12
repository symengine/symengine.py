from symengine.utilities import raises

from symengine import Integer, I


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


def test_is_conditions():
    i = Integer(-123)
    assert not i.is_zero
    assert not i.is_positive
    assert i.is_negative
    assert i.is_nonzero
    assert i.is_nonpositive
    assert not i.is_nonnegative
    assert not i.is_complex

    i = Integer(123)
    assert not i.is_zero
    assert i.is_positive
    assert not i.is_negative
    assert i.is_nonzero
    assert not i.is_nonpositive
    assert i.is_nonnegative
    assert not i.is_complex

    i = Integer(0)
    assert i.is_zero
    assert not i.is_positive
    assert not i.is_negative
    assert not i.is_nonzero
    assert i.is_nonpositive
    assert i.is_nonnegative
    assert not i.is_complex

    i = Integer(1) + I
    assert not i.is_zero
    assert not i.is_positive
    assert not i.is_negative
    assert not i.is_nonzero
    assert not i.is_nonpositive
    assert not i.is_nonnegative
    assert i.is_complex
