import ast
from pathlib import Path


def test_robotwin_client_propagates_required_cli_args_to_config():
    source = Path("evaluation/robotwin/eval_polict_client_openpi.py").read_text()

    assert "CLI_CONFIG_KEYS" in source
    for key in (
        "port",
        "save_root",
        "video_guidance_scale",
        "action_guidance_scale",
        "test_num",
        "robotwin_root",
    ):
        assert f'"{key}"' in source


def test_robotwin_client_override_parser_is_literal_only():
    source = Path("evaluation/robotwin/eval_polict_client_openpi.py").read_text()
    tree = ast.parse(source)

    assert "ast.literal_eval" in source
    assert "--overrides expects key/value pairs" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "eval"


def test_robotwin_eval_client_has_diagnostic_exceptions():
    source = Path("evaluation/robotwin/eval_polict_client_openpi.py").read_text()
    tree = ast.parse(source)

    assert "No Task: {task_name}" in source
    assert "unsupported action dimension" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None
