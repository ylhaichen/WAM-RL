from pathlib import Path


def test_server_flushes_async_saves_before_returning_artifact_paths():
    utils_source = Path("wan_va/utils/utils.py").read_text()
    server_source = Path("wan_va/wan_va_server.py").read_text()

    assert "def flush_async_saves" in utils_source
    assert "flush_async_saves" in server_source
    assert "strict_grpo_path" in server_source
    assert "strict_grpo_paths" in server_source


def test_server_debug_tensor_saves_can_be_disabled():
    server_source = Path("wan_va/wan_va_server.py").read_text()

    assert "save_server_debug_tensors" in server_source
    assert "latent_path = None" in server_source
    assert '"latent_path": latent_path or ""' in server_source
    assert "obs_data_{self.frame_st_id}.pt" in server_source


def test_server_can_gate_strict_grpo_capture_by_chunk():
    server_source = Path("wan_va/wan_va_server.py").read_text()

    assert "def _should_capture_strict_grpo_chunk" in server_source
    assert "strict_grpo_capture_chunk_stride" in server_source
    assert "strict_grpo_capture_max_chunks" in server_source
    assert "strict_grpo_capture_for_chunk" in server_source
    assert "capture_this_step = strict_grpo_capture_for_chunk" in server_source
