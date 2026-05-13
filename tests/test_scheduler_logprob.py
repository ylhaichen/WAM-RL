import torch

from wan_va.rl.scheduler_logprob import stochastic_flowmatch_step


class DummyScheduler:
    def step(self, model_output, timestep, sample, return_dict=False):
        return sample + model_output


def test_stochastic_flowmatch_step_is_reproducible_with_seeded_generator():
    sample = torch.zeros(1, 1, 1, 2, 1)
    model_output = torch.ones_like(sample) * 0.5
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)

    out_a = stochastic_flowmatch_step(
        scheduler=DummyScheduler(),
        model_output=model_output,
        timestep=torch.tensor(1.0),
        sample=sample,
        transition_std=0.1,
        generator=generator_a,
    )
    out_b = stochastic_flowmatch_step(
        scheduler=DummyScheduler(),
        model_output=model_output,
        timestep=torch.tensor(1.0),
        sample=sample,
        transition_std=0.1,
        generator=generator_b,
    )

    assert torch.allclose(out_a.next_state, out_b.next_state)
    assert torch.allclose(out_a.logprob_sum, out_b.logprob_sum)
    assert torch.allclose(out_a.transition_mean, torch.full_like(sample, 0.5))
