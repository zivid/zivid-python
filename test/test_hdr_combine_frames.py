def test_list_one_element(frame):
    import zivid
    from zivid.hdr import combine_frames

    frame_list = [frame]
    assert isinstance(frame_list, list)
    with combine_frames(frame_list) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.Frame)


def test_tuple_one_element(frame):
    import zivid
    from zivid.hdr import combine_frames

    frame_tuple = (frame,)
    assert isinstance(frame_tuple, tuple)
    with combine_frames(frame_tuple) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.Frame)


def test_list_five_elements(three_frames):
    import zivid
    from zivid.hdr import combine_frames

    frame_list = three_frames
    assert isinstance(frame_list, list)
    with combine_frames(frame_list) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.Frame)


def test_tuple_five_elements(three_frames):
    import zivid
    from zivid.hdr import combine_frames

    frame_tuple = tuple(three_frames)
    assert isinstance(frame_tuple, tuple)
    with combine_frames(frame_tuple) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.Frame)
