import pytest

from modconv.utils import format_size


@pytest.mark.parametrize("value,expected", [
    (0, "0 B"),
    (1, "1 B"),
    (1023, "1023 B"),
    (1024, "1.00 KB"),
    (1536, "1.50 KB"),
    (1024 ** 2, "1.00 MB"),
    (1024 ** 3, "1.00 GB"),
    (1024 ** 4, "1.00 TB"),
    (5 * 1024 ** 3 + 100, "5.00 GB"),
])
def test_format_size(value, expected):
    assert format_size(value) == expected
