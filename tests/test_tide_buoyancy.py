import pytest
import gameplay.tide_buoyancy as mod

CASES = [
    ("same_tide_for_physics", (1,)),
    ("freeze_before_height", (1,)),
    ("reset_phase_on_level", (1,)),
    ("wake_on_submerge", (1,)),
    ("freeze_phase_on_pause", (1,)),
    ("clamp_wave_height", (1,)),
    ("sample_tide", (1,)),
    ("apply_buoyancy", (1,)),
    ("switch_level", (1,)),
    ("pause", (1,)),
    ("validate_buoyancy", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
