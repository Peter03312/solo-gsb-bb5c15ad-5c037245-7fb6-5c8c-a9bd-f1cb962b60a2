"""统一余量放宽接口 POST /api/v1/diagnose/slack 的 HTTP 测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_slack_k_zero_returns_original_intervals():
    payload = {
        "reference": "A",
        "L": 5,
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 1, "max": 5},
        ],
    }
    resp = client.post("/api/v1/diagnose/slack", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["slack"] == 0
    assert data["limit"] == 5
    assert data["observations"] == [
        {"id": "o1", "source": "A", "target": "B", "min": 1.0, "max": 5.0}
    ]
    offsets = {x["node"]: x["offset"] for x in data["node_offsets"]}
    assert offsets["A"] == 0.0


def test_slack_contradiction_resolved_with_minimal_k():
    payload = {
        "reference": "A",
        "L": 5,
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
            {"id": "o2", "source": "B", "target": "A", "min": 3, "max": 4},
        ],
    }
    resp = client.post("/api/v1/diagnose/slack", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["slack"] == 2
    assert data["observations"] == [
        {"id": "o1", "source": "A", "target": "B", "min": -2.0, "max": 12.0},
        {"id": "o2", "source": "B", "target": "A", "min": 1.0, "max": 6.0},
    ]
    # 见证覆盖全部节点、参考节点为 0，并满足每条放宽后的区间。
    offsets = {x["node"]: x["offset"] for x in data["node_offsets"]}
    assert offsets["A"] == 0.0
    for obs in data["observations"]:
        diff = offsets[obs["target"]] - offsets[obs["source"]]
        assert obs["min"] <= diff <= obs["max"]


def test_slack_threshold_boundary():
    payload_obs = [
        {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
        {"id": "o2", "source": "B", "target": "A", "min": 3, "max": 4},
    ]
    below = client.post(
        "/api/v1/diagnose/slack",
        json={"reference": "A", "L": 1, "observations": payload_obs},
    ).json()
    assert below == {
        "status": "limit_exceeded",
        "name": None,
        "reference": "A",
        "limit": 1,
    }

    at = client.post(
        "/api/v1/diagnose/slack",
        json={"reference": "A", "L": 2, "observations": payload_obs},
    ).json()
    assert at["status"] == "ok"
    assert at["slack"] == 2


def test_slack_limit_exceeded_omits_intervals_and_witness():
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={
            "reference": "A",
            "L": 0,
            "observations": [
                {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
                {"id": "o2", "source": "B", "target": "A", "min": 3, "max": 4},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "limit_exceeded"
    assert "observations" not in data
    assert "node_offsets" not in data
    assert "slack" not in data


def test_slack_empty_observations_feasible():
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={"reference": "A", "nodes": ["A", "Z"], "L": 0, "observations": []},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["slack"] == 0
    assert data["observations"] == []
    assert data["node_offsets"] == [
        {"node": "A", "offset": 0.0},
        {"node": "Z", "offset": 0.0},
    ]


def test_slack_name_echoed():
    cases = (
        (
            "ok",
            {
                "reference": "A",
                "L": 3,
                "name": "run-1",
                "observations": [
                    {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 1}
                ],
            },
        ),
        (
            "limit_exceeded",
            {
                "reference": "A",
                "L": 0,
                "name": "run-2",
                "observations": [
                    {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
                    {"id": "o2", "source": "B", "target": "A", "min": 3, "max": 4},
                ],
            },
        ),
    )
    for status, body in cases:
        data = client.post("/api/v1/diagnose/slack", json=body).json()
        assert data["name"] == body["name"]
        assert data["status"] == status


def test_slack_duplicate_ids_rejected():
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={
            "reference": "A",
            "L": 3,
            "observations": [
                {"id": "x", "source": "A", "target": "B", "min": 0, "max": 1},
                {"id": "x", "source": "B", "target": "A", "min": 0, "max": 1},
            ],
        },
    )
    assert resp.status_code == 422
    assert "重复" in resp.text


def test_slack_invalid_interval_rejected():
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={
            "reference": "A",
            "L": 3,
            "observations": [
                {"id": "o1", "source": "A", "target": "B", "min": 5, "max": 1}
            ],
        },
    )
    assert resp.status_code == 422
    assert "区间非法" in resp.text


def test_slack_non_finite_bounds_rejected():
    # NaN/±inf 非合法 JSON，按现有用例以原始内容发送 NaN / Infinity 字面量。
    for raw_bound in ("NaN", "Infinity"):
        for bound_field in ("min", "max"):
            other = {"min": 0, "max": 1}
            other[bound_field] = raw_bound
            body = (
                '{"reference":"A","L":3,"observations":['
                f'{{"id":"o1","source":"A","target":"B",'
                f'"min":{other["min"]},"max":{other["max"]}}}'
                "]}"
            )
            resp = client.post(
                "/api/v1/diagnose/slack",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 422, body


@pytest.mark.parametrize("bad_limit", [-1, 1.5, "3", True, None])
def test_slack_limit_must_be_non_negative_integer(bad_limit):
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={"reference": "A", "L": bad_limit, "observations": []},
    )
    assert resp.status_code == 422


def test_slack_limit_required():
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={"reference": "A", "observations": []},
    )
    assert resp.status_code == 422


def test_slack_extra_field_rejected():
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={"reference": "A", "L": 1, "observations": [], "bogus": 1},
    )
    assert resp.status_code == 422


def test_slack_reference_must_be_in_nodes():
    resp = client.post(
        "/api/v1/diagnose/slack",
        json={
            "reference": "C",
            "nodes": ["A", "B"],
            "L": 1,
            "observations": [
                {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 1}
            ],
        },
    )
    assert resp.status_code == 422
