import pytest
from newspaper_ocr.registry import Registry


def test_register_and_get():
    reg = Registry("test")
    reg.register("foo", str)
    assert reg.get("foo") is str


def test_get_unknown_raises():
    reg = Registry("test")
    with pytest.raises(KeyError, match="Unknown test"):
        reg.get("bar")


def test_list_registered():
    reg = Registry("test")
    reg.register("a", int)
    reg.register("b", str)
    assert reg.list() == ["a", "b"]
