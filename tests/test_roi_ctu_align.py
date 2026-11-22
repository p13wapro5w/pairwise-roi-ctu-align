import pytest
import media.roi_ctu_align as mod

CASES = [
    ("ctu_align", (64, 2)),
    ("roi_mask_smooth", (1, 2)),
    ("qp_offset", (1, 2)),
    ("boundary_block", (1, 2)),
    ("bitrate_compare", (1, 2)),
    ("roi_recompute", (1, 2)),
    ("block_artifact", (1, 2)),
    ("validation", (1, 2)),
    ("mask_dilate", (1, 2)),
    ("ctu_grid", (1, 2)),
    ("rate_control", (1, 2)),
    ("fallback_qp", (1, 2)),
    ("bitrate_metric", (1.0, 2.0)),
    ("batch_guard", (1, 2)),
    ("edge_guard", (1, 2)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
