from wan_va.rl.rollout_worker import build_grouped_rollout_plan


def test_build_grouped_rollout_plan_assigns_tasks_ports_and_sampling_seeds():
    plan = build_grouped_rollout_plan(
        task_names=["hanging_mug", "open_microwave", "turn_switch"],
        group_size=2,
        groups_per_task=2,
        start_seed=10000,
        prompt_index=0,
        num_gpus=2,
        start_port=30156,
    )

    assert plan.total_context_groups == 2
    assert plan.total_rollouts == 12
    assert len(plan.batches) == 4
    assert plan.batches[0].group_index == 0
    assert plan.batches[0].sample_idx == 0
    assert plan.batches[0].env_seed == 10000
    assert plan.batches[0].sampling_seed == 10000000000
    assert [(task.task_name, task.gpu_id, task.port) for task in plan.batches[0].tasks] == [
        ("hanging_mug", 0, 30156),
        ("open_microwave", 1, 30157),
        ("turn_switch", 0, 30156),
    ]
    assert plan.batches[-1].group_index == 1
    assert plan.batches[-1].sample_idx == 1
    assert plan.batches[-1].sampling_seed == 10000000003
