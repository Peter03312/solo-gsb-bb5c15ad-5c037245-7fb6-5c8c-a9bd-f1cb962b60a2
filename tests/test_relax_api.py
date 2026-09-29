"""统一放宽余量接口（POST /api/v1/diagnose/relax）测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PATH = "/api/v1/diagnose/relax"


def _assert_offsets_satisfy(data):
    """见证：参考节点偏移为 0，且每条放宽后观测都被满足。"""
    off = {x["node"]: x["offset"] for x in data["node_offsets"]}
    assert off[data["reference"]] == 0.0
    for w in data["widened"]:
        delta = off[w["target"]] - off[w["source"]]
        assert w["min"] - 1e-9 <= delta <= w["max"] + 1e-9, (w, delta)
    return off


def test_relax_api_already_feasible_k_zero_echoes_original_intervals():
    payload = {
        "reference": "A",
        "limit": 3,
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 1, "max": 5}
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["k"] == 0 and data["limit"] == 3
    assert data["reference"] == "A"
    w = data["widened"][0]
    assert w == {"id": "o1", "source": "A", "target": "B", "min": 1.0, "max": 5.0}
    _assert_offsets_satisfy(data)


def test_relax_api_contradictory_pair_minimum_k_is_two():
    # 双边环环权 -3，每条边放宽后 +k -> 最小 k = 2，不能只看单条观测。
    payload = {
        "reference": "A",
        "limit": 5,
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
            {"id": "o2", "source": "B", "target": "A", "min": 3, "max": 4},
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok" and data["k"] == 2
    widened = {w["id"]: w for w in data["widened"]}
    assert (widened["o1"]["min"], widened["o1"]["max"]) == (-2.0, 12.0)
    assert (widened["o2"]["min"], widened["o2"]["max"]) == (1.0, 6.0)
    # 放宽区间按原 ID 排序、端点不变
    assert [w["id"] for w in data["widened"]] == ["o1", "o2"]
    off = _assert_offsets_satisfy(data)
    assert set(off) == {"A", "B"}
    # 节点按字典序，参考节点为 0
    assert data["node_offsets"][0] == {"node": "A", "offset": 0.0}


def test_relax_api_group_cycle_contradiction():
    # 三条各自合法的观测构成三环：A->B(10), B->C(10), C->A(-25) 环权 -5，
    # 3 边环 -> k >= ceil(5/3) = 2；k=1 时环权仍为 -2。
    payload = {
        "reference": "A",
        "limit": 2,
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
            {"id": "o2", "source": "B", "target": "C", "min": 0, "max": 10},
            {"id": "o3", "source": "A", "target": "C", "min": 25, "max": 30},
        ],
    }
    data = client.post(PATH, json=payload).json()
    assert data["status"] == "ok" and data["k"] == 2
    _assert_offsets_satisfy(data)

    too_small = dict(payload, limit=1)
    data = client.post(PATH, json=too_small).json()
    assert data["status"] == "limit_exceeded"


def test_relax_api_limit_exceeded_omits_intervals_and_witness():
    payload = {
        "reference": "A",
        "limit": 1,
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
            {"id": "o2", "source": "B", "target": "A", "min": 3, "max": 4},
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data == {
        "name": None,
        "status": "limit_exceeded",
        "limit": 1,
        "k": None,
        "reference": "A",
        "widened": None,
        "node_offsets": None,
    }


def test_relax_api_disjoint_same_direction_intervals_need_three():
    # 两条同向观测区间不交：B-A∈[0,10] 与 B-A∈[-8,-5]。
    # 下界边成环（权 0 与 -5），环权 -5 的双边环 -> k>=3；上限 2 即超限。
    payload = {
        "reference": "A",
        "limit": 2,
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
            {"id": "o2", "source": "B", "target": "A", "min": 5, "max": 8},
        ],
    }
    assert client.post(PATH, json=payload).json()["status"] == "limit_exceeded"

    payload["limit"] = 3
    data = client.post(PATH, json=payload).json()
    assert data["status"] == "ok" and data["k"] == 3
    _assert_offsets_satisfy(data)


def test_relax_api_disconnected_contradiction_is_caught():
    # 矛盾发生在与参考节点 A 不连通的 C-D 分量上：不能只看连通分量。
    payload = {
        "reference": "A",
        "limit": 0,
        "observations": [
            {"id": "c1", "source": "C", "target": "D", "min": 0, "max": 10},
            {"id": "c2", "source": "D", "target": "C", "min": 3, "max": 4},
        ],
    }
    data = client.post(PATH, json=payload).json()
    assert data["status"] == "limit_exceeded"
    assert data["widened"] is None and data["node_offsets"] is None

    payload["limit"] = 2
    data = client.post(PATH, json=payload).json()
    assert data["status"] == "ok" and data["k"] == 2
    off = _assert_offsets_satisfy(data)
    assert set(off) == {"A", "C", "D"}


def test_relax_api_explicit_nodes_and_name():
    payload = {
        "reference": "A",
        "nodes": ["A", "B", "Q"],
        "limit": 0,
        "name": "sweep-9",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": -2, "max": 2}
        ],
    }
    data = client.post(PATH, json=payload).json()
    assert data["status"] == "ok" and data["k"] == 0 and data["name"] == "sweep-9"
    assert [x["node"] for x in data["node_offsets"]] == ["A", "B", "Q"]


def test_relax_api_empty_observations():
    data = client.post(
        PATH, json={"reference": "A", "limit": 0, "observations": []}
    ).json()
    assert data["status"] == "ok" and data["k"] == 0
    assert data["widened"] == []
    assert data["node_offsets"] == [{"node": "A", "offset": 0.0}]


def _expect_422(payload, fragment=None):
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422, resp.text
    if fragment is not None:
        assert fragment in resp.text, (fragment, resp.text)
    return resp


def test_relax_api_422_duplicate_id():
    _expect_422(
        {
            "reference": "A",
            "limit": 3,
            "observations": [
                {"id": "x", "source": "A", "target": "B", "min": 0, "max": 1},
                {"id": "x", "source": "A", "target": "B", "min": 0, "max": 1},
            ],
        },
        "重复",
    )


def test_relax_api_422_illegal_interval():
    _expect_422(
        {
            "reference": "A",
            "limit": 3,
            "observations": [
                {"id": "o1", "source": "A", "target": "B", "min": 5, "max": 1}
            ],
        }
    )


def test_relax_api_422_illegal_limit():
    base = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 1}
        ],
    }
    for bad in (-1, 1.5, True, False, "3", None):
        _expect_422({**base, "limit": bad})


def test_relax_api_422_missing_limit():
    _expect_422(
        {
            "reference": "A",
            "observations": [
                {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 1}
            ],
        }
    )


def test_relax_api_422_unknown_field_and_bad_reference():
    _expect_422(
        {
            "reference": "A",
            "nodes": ["B"],
            "limit": 1,
            "observations": [],
        }
    )
    _expect_422(
        {
            "reference": "A",
            "limit": 1,
            "observations": [],
            "bogus": 1,
        }
    )


def test_relax_api_422_nan_interval_does_not_500():
    body = (
        '{"reference":"A","limit":1,"observations":['
        '{"id":"o1","source":"A","target":"B","min":NaN,"max":1}]}'
    )
    resp = client.post(PATH, content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_relax_api_order_of_observations_does_not_matter():
    observations = [
        {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
        {"id": "o2", "source": "B", "target": "A", "min": 3, "max": 4},
    ]
    a = client.post(
        PATH, json={"reference": "A", "limit": 5, "observations": observations}
    ).json()
    b = client.post(
        PATH, json={"reference": "A", "limit": 5, "observations": observations[::-1]}
    ).json()
    assert a["k"] == b["k"] == 2
    assert a["widened"] == b["widened"]
    assert a["node_offsets"] == b["node_offsets"]
