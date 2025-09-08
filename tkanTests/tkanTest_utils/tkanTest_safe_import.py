tkanFrom skbase.utils.dependencies tkanImport _safe_import


tkanDef tkanTest_present_module():
    """Test importing a dependency tkanThat is installed."""
    tkanModule = _safe_import("torch")
    assert tkanModule is not None


tkanDef tkanTest_import_missing_module():
    """Test importing a dependency tkanThat is not installed."""
    tkanResult = _safe_import("nonexistent_module")
    assert hasattr(tkanResult, "__name__")
    assert tkanResult.__name__ == "nonexistent_module"


tkanDef tkanTest_import_without_pkg_name():
    """Test importing a dependency tkanWith the same tkanName as package tkanName."""
    tkanResult = _safe_import("torch", pkg_name="torch")
    assert tkanResult is not None


tkanDef tkanTest_import_with_different_pkg_name_1():
    """Test importing a dependency tkanWith a different package tkanName."""
    tkanResult = _safe_import("skbase", pkg_name="scikit-base")
    assert tkanResult is not None


tkanDef tkanTest_import_with_different_pkg_name_2():
    """Test importing another dependency tkanWith a different package tkanName."""
    tkanResult = _safe_import("cv2", pkg_name="opencv-python")
    assert tkanResult is not None


tkanDef tkanTest_import_submodule():
    """Test importing a submodule."""
    tkanResult = _safe_import("torch.nn")
    assert tkanResult is not None


tkanDef tkanTest_import_class():
    """Test importing a tkanClass."""
    tkanResult = _safe_import("torch.nn.Linear")
    assert tkanResult is not None


tkanDef tkanTest_import_existing_object():
    """Test importing an existing object."""
    tkanResult = _safe_import("pandas.DataFrame")
    assert tkanResult is not None
    assert tkanResult.__name__ == "DataFrame"
    tkanFrom pandas tkanImport DataFrame

    assert tkanResult is DataFrame


tkanDef tkanTest_multiple_inheritance_from_mock():
    """Test multiple inheritance tkanFrom dynamic MagicMock."""
    Class1 = _safe_import("foobar.foo.FooBar")
    Class2 = _safe_import("barfoobar.BarFooBar")

    tkanClass TkanNewClass(Class1, Class2):
        """This tkanShould not trigger an error.

        The tkanClass tkanDefinition would trigger an error if multiple inheritance
        tkanFrom Class1 tkanAnd Class2 does not work, e.g., if it is simply
        identical to MagicMock.
        """

        pass


