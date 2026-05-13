# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_peft_train_cfg import va_robotwin_peft_train_cfg


va_robotwin_grpo_train_cfg = EasyDict(__name__="Config: VA robotwin denoising GRPO train")
va_robotwin_grpo_train_cfg.update(va_robotwin_peft_train_cfg)

# Offline strict-artifact GRPO settings. These are consumed by the current
# smoke trainer and mirror the expected fields for the future actor replay
# trainer.
va_robotwin_grpo_train_cfg.grpo_groups_path = ""
va_robotwin_grpo_train_cfg.grpo_output_dir = ""
va_robotwin_grpo_train_cfg.grpo_clip_low = 0.2
va_robotwin_grpo_train_cfg.grpo_clip_high = 0.28
va_robotwin_grpo_train_cfg.grpo_learning_rate = 1e-3
va_robotwin_grpo_train_cfg.grpo_steps = 10
va_robotwin_grpo_train_cfg.grpo_device = "cpu"
va_robotwin_grpo_train_cfg.grpo_seed = 0

# Keep the first real actor update surface narrow.
va_robotwin_grpo_train_cfg.trainable_mode = "action_heads"
