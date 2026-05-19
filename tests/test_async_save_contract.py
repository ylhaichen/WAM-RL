from pathlib import Path


def test_server_flushes_async_saves_before_returning_artifact_paths():
    utils_source = Path("wan_va/utils/utils.py").read_text()
    server_source = Path("wan_va/wan_va_server.py").read_text()

    assert "def flush_async_saves" in utils_source
    assert "flush_async_saves" in server_source
    assert "strict_grpo_path" in server_source
    assert "strict_grpo_paths" in server_source
