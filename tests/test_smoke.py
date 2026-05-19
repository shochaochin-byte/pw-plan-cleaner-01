from cleaner.geometry_parser import VectorPrimitive


def test_vector_primitive_fields():
    prim = VectorPrimitive(kind="line", bbox=(0.0, 0.0, 1.0, 1.0), width=1.0, angle=0.0, length=1.414, page=1)
    assert prim.kind == "line"
    assert prim.page == 1
