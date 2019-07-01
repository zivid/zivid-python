def test_timestamp():
    import datetime
    import zivid

    info = zivid.FrameInfo()
    timestamp = info.timestamp
    assert timestamp
    assert isinstance(timestamp, datetime.datetime)
