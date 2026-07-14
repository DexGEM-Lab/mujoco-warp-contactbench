def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires the accepted local Lance dataset")
    config.addinivalue_line("markers", "mjx_warp: exercises the MJX-Warp runtime")
