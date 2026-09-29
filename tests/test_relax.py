"""统一放宽余量求解器测试（relax_observations）。"""

from app.solver import Observation, relax_observations


def _obs(id_, source, target, lo, hi):
    return Observation(id=id_, source=source, target=target, min=lo, max=hi)


def _assert_witness(result):
    """见证：参考节点偏移为 0，且全部放宽后观测都被满足。"""
    by_id = {w["id"]: w for w in result["widened"]}
    off = {x["node"]: x["offset"] for x in result["node_offsets"]}
    ref = result["reference"]
    assert off[ref] == 0.0
    for wid in by_id.values():
        delta = off[wid["target"]] - off[wid["source"]]
        assert wid["min"] - 1e-9 <= delta <= wid["max"] + 1e-9, (wid, delta)
    return off, by_id


def test_relax_already_feasible_k_zero_returns_original_intervals():
    result = relax_observations(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 1, 5)],
        limit=3,
    )
    assert result["status"] == "ok"
    assert result["k"] == 0
    assert result["limit"] == 3
    (w,) = result["widened"]
    assert w["min"] == 1.0 and w["max"] == 5.0  # k=0 原区间照返
    _assert_witness(result)


def test_relax_two_edge_contradiction_requires_two():
    # o1: B-A∈[0,10]，o2: A-B∈[3,4]；双边环环权 -3，每边加 k -> k>=2
    obs = [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 3, 4)]
    assert relax_observations(["A", "B"], "A", obs, 1)["status"] == "limit_exceeded"
    result = relax_observations(["A", "B"], "A", obs, 2)
    assert result["status"] == "ok" and result["k"] == 2
    w = {x["id"]: x for x in result["widened"]}
    assert (w["o1"]["min"], w["o1"]["max"]) == (-2.0, 12.0)
    assert (w["o2"]["min"], w["o2"]["max"]) == (1.0, 6.0)
    _assert_witness(result)


def test_relax_group_contradiction_needs_cycle_not_single_observation():
    # 三条观测各自 min<=max，单看都合法；矛盾只存在于三环：
    # A->B(10), B->C(10), C->A(-25) 环权 -5，3 边 -> k>=ceil(5/3)=2
    obs = [
        _obs("o1", "A", "B", 0, 10),
        _obs("o2", "B", "C", 0, 10),
        _obs("o3", "A", "C", 25, 30),
    ]
    assert relax_observations(["A", "B", "C"], "A", obs, 1)["status"] == (
        "limit_exceeded"
    )
    result = relax_observations(["A", "B", "C"], "A", obs, 2)
    assert result["status"] == "ok" and result["k"] == 2
    _assert_witness(result)


def test_relax_self_loop_infeasible_interval():
    # 自环 x_A - x_A ∈ [1, 0]：单边环权 -1，k>=1
    result = relax_observations(["A"], "A", [_obs("s", "A", "A", 1, 0)], 2)
    assert result["status"] == "ok" and result["k"] == 1
    (w,) = result["widened"]
    assert (w["min"], w["max"]) == (0.0, 1.0)
    _assert_witness(result)
    assert relax_observations(["A"], "A", [_obs("s", "A", "A", 1, 0)], 0)[
        "status"
    ] == "limit_exceeded"


def test_relax_disconnected_component_contradiction_is_detected():
    # 矛盾分量与参考节点不连通：全图负环检测不能漏
    obs = [
        _obs("a1", "A", "B", 0, 10),
        _obs("a2", "B", "A", 3, 4),
        _obs("c1", "C", "D", 0, 10),
        _obs("c2", "D", "C", 3, 4),
    ]
    result = relax_observations(["A", "B", "C", "D"], "A", obs, 2)
    assert result["status"] == "ok" and result["k"] == 2
    off, _ = _assert_witness(result)
    assert set(off) == {"A", "B", "C", "D"}  # 见证覆盖全节点


def test_relax_limit_exceeded_omits_intervals_and_witness():
    result = relax_observations(
        ["A", "B"],
        "A",
        [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 3, 4)],
        1,
    )
    assert result == {
        "status": "limit_exceeded",
        "limit": 1,
        "reference": "A",
        "required_gt_limit": True,
    }


def test_relax_widened_sorted_by_original_id_independent_of_submission():
    obs = [_obs("z", "A", "B", 0, 10), _obs("a", "B", "A", 3, 4)]
    result = relax_observations(["A", "B"], "A", obs, 5)
    assert [w["id"] for w in result["widened"]] == ["a", "z"]
    # 端点与原观测一致
    w = {x["id"]: x for x in result["widened"]}
    assert (w["z"]["source"], w["z"]["target"]) == ("A", "B")
    assert (w["a"]["source"], w["a"]["target"]) == ("B", "A")


def test_relax_does_not_mutate_input_observations():
    obs = [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 3, 4)]
    relax_observations(["A", "B"], "A", obs, 5)
    assert (obs[0].min, obs[0].max) == (0, 10)
    assert (obs[1].min, obs[1].max) == (3, 4)


def test_relax_empty_observations_k_zero():
    result = relax_observations(["A", "B"], "A", [], 0)
    assert result["status"] == "ok" and result["k"] == 0
    assert result["widened"] == []
    off = {x["node"]: x["offset"] for x in result["node_offsets"]}
    assert off == {"A": 0.0, "B": 0.0}


def test_relax_fractional_cycle_rounds_up():
    # 两个同向区间不交：B-A∈[0,10] 与 B-A∈[-8,-5]（o2 记为 A-B∈[5,8]）。
    # 下界边环：o1 下界 B->A 权 0，o2 下界 A->B 权 -5 -> 环权 -5 的双边环，
    # 2k >= 5 -> k>=ceil(2.5)=3；k=2 时环权 -1 仍矛盾。
    obs = [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 5, 8)]
    assert relax_observations(["A", "B"], "A", obs, 2)["status"] == (
        "limit_exceeded"
    )
    result = relax_observations(["A", "B"], "A", obs, 3)
    assert result["k"] == 3
    _assert_witness(result)


def test_relax_single_observation_never_needs_widening():
    # 任何单条合法观测（min<=max）都是可行系统：不能只根据单条观测判断矛盾
    for lo, hi in [(100, 200), (-5, 5), (0, 0), (-10, -10)]:
        result = relax_observations(
            ["A", "B"], "A", [_obs("o1", "A", "B", lo, hi)], 0
        )
        assert result["status"] == "ok" and result["k"] == 0, (lo, hi)


def test_relax_node_offsets_cover_all_nodes_sorted():
    obs = [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 3, 4)]
    result = relax_observations(["A", "B", "C"], "A", obs, 5)
    assert [x["node"] for x in result["node_offsets"]] == ["A", "B", "C"]
    assert result["node_offsets"][0]["offset"] == 0.0
