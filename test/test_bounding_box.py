import zivid


def test_bounding_box_basics():

    bb = zivid.BoundingBox(x=10, y=20, width=300, height=400)
    assert bb.x == 10
    assert bb.y == 20
    assert bb.width == 300
    assert bb.height == 400
    assert str(bb) == "{ x: 10, y: 20, width: 300, height: 400 }"

    bb.x = 15
    bb.y = 25
    bb.width = 350
    bb.height = 450
    assert bb.x == 15
    assert bb.y == 25
    assert bb.width == 350
    assert bb.height == 450
    assert str(bb) == "{ x: 15, y: 25, width: 350, height: 450 }"
