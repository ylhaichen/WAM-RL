# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from copy import deepcopy

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_peft_train_cfg = deepcopy(va_robotwin_train_cfg)
va_robotwin_peft_train_cfg.__name__ = 'Config: VA robotwin PEFT train'

# First conservative update surface for Phase 3. This intentionally avoids
# broad full-model updates while we validate train/save/evaluate wiring.
va_robotwin_peft_train_cfg.trainable_mode = 'action_heads'
va_robotwin_peft_train_cfg.trainable_param_patterns = [
    'action_embedder',
    'condition_embedder_action',
    'action_proj_out',
]
va_robotwin_peft_train_cfg.learning_rate = 5e-5
va_robotwin_peft_train_cfg.weight_decay = 0.01
