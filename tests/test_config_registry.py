from pathlib import Path


def test_libero_configs_are_registered():
    registry = Path("wan_va/configs/__init__.py").read_text(encoding="utf-8")

    assert "from .va_libero_cfg import va_libero_cfg" in registry
    assert "from .va_libero_train_cfg import va_libero_train_cfg" in registry
    assert "from .va_libero_i2va import va_libero_i2va_cfg" in registry
    assert "'libero': va_libero_cfg" in registry
    assert "'libero_train': va_libero_train_cfg" in registry
    assert "'libero_i2av': va_libero_i2va_cfg" in registry


def test_robotwin_grpo_config_is_registered():
    registry = Path("wan_va/configs/__init__.py").read_text(encoding="utf-8")

    assert "from .va_robotwin_grpo_train_cfg import va_robotwin_grpo_train_cfg" in registry
    assert "'robotwin_grpo_train': va_robotwin_grpo_train_cfg" in registry
