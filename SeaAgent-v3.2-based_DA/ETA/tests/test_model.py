import pytest

torch = pytest.importorskip("torch")

from eta.model import build_model


def _config(variant: str):
    return {
        "model": {
            "variant": variant,
            "input_dim": 8,
            "tokens_per_frame": 2,
            "hidden_dim": 16,
            "stages": 2,
            "layers_per_stage": 2,
            "attention_heads": 4,
            "ffn_dim": 32,
            "dropout": 0.0,
            "heads_hidden": 8,
        }
    }


@pytest.mark.parametrize(
    "variant,has_completeness",
    [
        ("eta", True),
        ("ca_only", True),
        ("ca_sh", False),
        ("mlp_cur", True),
        ("mlp_hist", True),
        ("sa_dh", True),
    ],
)
def test_all_variants_return_expected_head_shapes(variant: str, has_completeness: bool):
    model = build_model(_config(variant))
    candidate = torch.randn(3, 8)
    history = torch.randn(3, 3, 8)
    history_mask = torch.tensor(
        [[False, False, False], [True, False, False], [True, True, True]]
    )
    admission, completeness = model(candidate, history, history_mask)
    assert admission.shape == (3, 2)
    assert torch.isfinite(admission).all()
    if has_completeness:
        assert completeness.shape == (3, 2)
        assert torch.isfinite(completeness).all()
    else:
        assert completeness is None
