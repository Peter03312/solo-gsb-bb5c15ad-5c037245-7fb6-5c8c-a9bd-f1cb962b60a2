"""统一余量放宽求解器（find_uniform_slack）算法层测试。"""

import pytest

from app.solver import Observation, find_uniform_slack


def _obs(id_, source, target, min_, max_):
    return Observation(id=id_, source=source, target=target, min=min_, max=max_)


def test_already_feasible_returns_zero_and_original_intervals():
    observations = [_obs("o1", "A", "B", 1, 5)]
    result = find_uniform_slack(["A", "B"], "A", observations, limit=5)
    assert result["status"] == "ok"
    assert result["slack"] == 0
    assert result["observations"] == [
        {"id": "o1", "source": "A", "target": "B", "min": 1.0, "max": 5.0}
    ]


def test_contradictory_pair_needs_half_gap():
    # B-A ∈ [0,10] 与 A-B ∈ [3,4]（即 B-A ∈ [-4,-3]）矛盾：
    # 放宽后区间 [-k,10+k] 与 [-4-k,-3+k] 相交需要 2k >= 3，最小整数 k=2。
    observations = [
        _obs("o1", "A", "B", 0, 10),
        _obs("o2", "B", "A", 3, 4),
    ]
    result = find_uniform_slack(["A", "B"], "A", observations, limit=5)
    assert result["status"] == "ok"
    assert result["slack"] == 2
    assert result["observations"] == [
        {"id": "o1", "source": "A", "target": "B", "min": -2.0, "max": 12.0},
        {"id": "o2", "source": "B", "target": "A", "min": 1.0, "max": 6.0},
    ]


def test_witness_satisfies_every_relaxed_interval_with_reference_zero():
    observations = [
        _obs("o1", "A", "B", 0, 10),
        _obs("o2", "B", "A", 3, 4),
    ]
    result = find_uniform_slack(["A", "B"], "A", observations, limit=5)
    offsets = {item["node"]: item["offset"] for item in result["node_offsets"]}
    assert offsets == {"A": 0.0, "B": -1.0}
    by_id = {o["id"]: o for o in result["observations"]}
    for obs in observations:
        relaxed = by_id[obs.id]
        diff = offsets[obs.target] - offsets[obs.source]
        assert relaxed["min"] - 1e-9 <= diff <= relaxed["max"] + 1e-9


def test_three_cycle_requires_round_up_of_fraction():
    # 反向环权重和 3k - 20 >= 0 ⇒ k >= 20/3，最小整数 k=7（非四舍五入，向上取整）。
    observations = [
        _obs("o1", "A", "B", 10, 20),
        _obs("o2", "B", "C", 10, 20),
        _obs("o3", "C", "A", 0, 5),
    ]
    result = find_uniform_slack(["A", "B", "C"], "A", observations, limit=100)
    assert result["status"] == "ok"
    assert result["slack"] == 7
    # 上限恰为阈值时可解；少 1 不可解。
    assert (
        find_uniform_slack(["A", "B", "C"], "A", observations, limit=7)["slack"] == 7
    )
    below = find_uniform_slack(["A", "B", "C"], "A", observations, limit=6)
    assert below["status"] == "limit_exceeded"
    assert below == {
        "status": "limit_exceeded",
        "reference": "A",
        "limit": 6,
    }


def test_limit_exceeded_carries_no_intervals_or_witness():
    observations = [
        _obs("o1", "A", "B", 0, 10),
        _obs("o2", "B", "A", 3, 4),
    ]
    result = find_uniform_slack(["A", "B"], "A", observations, limit=1)
    assert result["status"] == "limit_exceeded"
    assert result["limit"] == 1
    assert "observations" not in result
    assert "node_offsets" not in result
    assert "slack" not in result


def test_empty_observations_feasible_at_zero():
    result = find_uniform_slack(["A"], "A", [], limit=0)
    assert result["status"] == "ok"
    assert result["slack"] == 0
    assert result["observations"] == []
    assert result["node_offsets"] == [{"node": "A", "offset": 0.0}]


def test_isolated_nodes_included_in_witness():
    # 孤岛节点不与参考节点相连：零超源可行势平移后其取值任意，
    # 只保证参考节点偏移为 0、见证覆盖全部节点。
    observations = [_obs("o1", "A", "B", 1, 5)]
    result = find_uniform_slack(["A", "B", "Z"], "A", observations, limit=0)
    offsets = {item["node"]: item["offset"] for item in result["node_offsets"]}
    assert offsets["A"] == 0.0
    assert set(offsets) == {"A", "B", "Z"}


def test_single_infeasible_observation_self_loop():
    # 自环 B-B ∈ [1, 2] 要求 0 ∈ [1-k, 2+k]，k >= 1。
    observations = [_obs("solo", "B", "B", 1, 2)]
    assert (
        find_uniform_slack(["B"], "B", observations, limit=0)["status"]
        == "limit_exceeded"
    )
    result = find_uniform_slack(["B"], "B", observations, limit=3)
    assert result["slack"] == 1
    assert result["observations"][0]["min"] == 0.0
    assert result["observations"][0]["max"] == 3.0


def test_input_observations_not_mutated():
    observations = [
        _obs("o1", "A", "B", 0, 10),
        _obs("o2", "B", "A", 3, 4),
    ]
    originals = [(o.id, o.min, o.max) for o in observations]
    find_uniform_slack(["A", "B"], "A", observations, limit=5)
    assert [(o.id, o.min, o.max) for o in observations] == originals


def test_minimality_against_bruteforce_on_random_systems():
    # 随机系统：二分答案必须与逐 k 线性扫描的首个可行值一致（可行性是整组判定）。
    import random

    rng = random.Random(20260929)
    nodes = ["A", "B", "C", "D"]
    for _ in range(40):
        observations = []
        for i in range(rng.randint(1, 6)):
            s, t = rng.sample(nodes, 2) if rng.random() < 0.8 else (rng.choice(nodes),) * 2
            lo = rng.randint(-5, 5)
            observations.append(
                _obs(f"o{i}", s, t, lo, lo + rng.randint(0, 6))
            )
        limit = rng.randint(0, 15)
        result = find_uniform_slack(nodes, "A", observations, limit=limit)

        def feasible(k):
            return (
                find_uniform_slack(nodes, "A", observations, limit=k)["status"]
                == "ok"
            )

        expected = next((k for k in range(limit + 1) if feasible(k)), None)
        if expected is None:
            assert result["status"] == "limit_exceeded"
        else:
            assert result["status"] == "ok"
            assert result["slack"] == expected
            # 见证在最优 k 处满足全部放宽区间，参考节点为 0。
            offsets = {x["node"]: x["offset"] for x in result["node_offsets"]}
            assert offsets["A"] == 0.0
            by_id = {o["id"]: o for o in result["observations"]}
            for obs in observations:
                r = by_id[obs.id]
                diff = offsets[obs.target] - offsets[obs.source]
                assert r["min"] - 1e-9 <= diff <= r["max"] + 1e-9


def test_huge_limit_converges_logarithmically():
    observations = [
        _obs("o1", "A", "B", 0, 10),
        _obs("o2", "B", "A", 3, 4),
    ]
    result = find_uniform_slack(["A", "B"], "A", observations, limit=10**18)
    assert result["slack"] == 2


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_monotonicity(k):
    observations = [
        _obs("o1", "A", "B", 0, 10),
        _obs("o2", "B", "A", 3, 4),
    ]
    at_k = find_uniform_slack(["A", "B"], "A", observations, limit=k)
    if k < 2:
        assert at_k["status"] == "limit_exceeded"
    else:
        assert at_k["status"] == "ok"
