def test_wan_va_package_import_is_lightweight():
    import wan_va

    assert "rl" in wan_va.__all__
