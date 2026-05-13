from pathlib import Path


def test_robotwin_client_propagates_required_cli_args_to_config():
    source = Path("evaluation/robotwin/eval_polict_client_openpi.py").read_text()

    assert "CLI_CONFIG_KEYS" in source
    for key in (
        "save_root",
        "video_guidance_scale",
        "action_guidance_scale",
        "test_num",
        "robotwin_root",
    ):
        assert f'"{key}"' in source
