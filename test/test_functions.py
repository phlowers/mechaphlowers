from mechaphlowers import welcome


def test_welcome() -> None:
    assert welcome() == "Welcome!"
