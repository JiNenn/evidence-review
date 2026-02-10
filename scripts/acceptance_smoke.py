#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    extra_headers: dict | None = None,
) -> dict | list:
    body = None
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)


def http_put_bytes(url: str, payload: bytes, content_type: str) -> None:
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": content_type},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=20):
        return


def http_get_status(url: str) -> int:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        _ = resp.read(1)
        return int(resp.status)


def poll(fn, timeout_sec: int = 30, interval_sec: float = 1.0):
    started = time.time()
    while True:
        value = fn()
        if value:
            return value
        if time.time() - started > timeout_sec:
            raise TimeoutError("poll timeout")
        time.sleep(interval_sec)


def assert_true(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def auth_headers(api_base: str, auth_user: str | None, auth_password: str | None) -> dict:
    if not auth_user:
        return {}
    token = http_json(
        "POST",
        f"{api_base}/auth/token",
        {"username": auth_user, "password": auth_password or ""},
    )
    return {"Authorization": f"Bearer {token['access_token']}"}


def test_normal_and_compat(api_base: str, headers: dict) -> tuple[str, int]:
    run = http_json(
        "POST",
        f"{api_base}/runs",
        {"task_prompt": "acceptance normal", "metadata": {"from": "script"}},
        headers,
    )
    run_id = run["id"]

    presign = http_json(
        "POST",
        f"{api_base}/runs/{run_id}/sources/presign-put",
        {"filename": "acceptance.txt", "content_type": "text/plain"},
        headers,
    )
    assert_true("source_doc_id" in presign, "presign response should include source_doc_id")
    assert_true(str(presign["source_doc_id"]) in presign["object_key"], "object key should include source_doc_id")
    presigned_source = http_json(
        "GET",
        f"{api_base}/sources/{presign['source_doc_id']}/presign-get",
        None,
        headers,
    )
    assert_true(
        presigned_source.get("object_key") == presign["object_key"],
        "source_doc should exist at presign stage and keep same object_key",
    )
    http_put_bytes(
        presign["url"],
        "この文書はレビュー論点抽出の根拠です。\n改善対象の差分を示します。".encode("utf-8"),
        "text/plain",
    )

    http_json(
        "POST",
        f"{api_base}/runs/{run_id}/sources/ingest",
        {
            "source_doc_id": presign["source_doc_id"],
            "title": "acceptance.txt",
            "content_type": "text/plain",
        },
        headers,
    )
    time.sleep(2)
    http_json("POST", f"{api_base}/runs/{run_id}/pipeline", {}, headers)

    issues = poll(lambda: http_json("GET", f"{api_base}/runs/{run_id}/issues", None, headers), timeout_sec=40)
    assert_true(isinstance(issues, list) and len(issues) > 0, "issues should exist")
    assert_true(all(issue.get("evidence_count", 0) >= 1 for issue in issues), "all issues must have evidence")
    assert_true(all(issue.get("status") != "hidden" for issue in issues), "hidden issues must not be listed")
    issues_all = http_json("GET", f"{api_base}/runs/{run_id}/issues?include_hidden=true", None, headers)
    assert_true(isinstance(issues_all, list), "issues include_hidden should return list")
    assert_true(len(issues_all) >= len(issues), "include_hidden should return at least as many issues")

    checked_source_docs: set[str] = set()
    for issue in issues:
        issue_detail = http_json("GET", f"{api_base}/issues/{issue['id']}", None, headers)
        evidences = issue_detail.get("evidences", [])
        assert_true(len(evidences) >= 1, "issue detail should include evidences")

        for evidence in evidences:
            assert_true(bool(evidence.get("loc")), "evidence loc must exist")
            assert_true(bool(evidence.get("chunk_loc")), "evidence chunk_loc must exist")
            assert_true(bool(evidence.get("source_doc_id")), "evidence source_doc_id must exist")
            assert_true(bool(evidence.get("chunk_id")), "evidence chunk_id must exist")
            if evidence.get("citation_span") is not None:
                assert_true(isinstance(evidence.get("citation_span"), dict), "citation_span must be an object")

            citation = http_json("GET", f"{api_base}/citations/{evidence['citation_id']}", None, headers)
            assert_true(citation.get("source_doc_id") == evidence.get("source_doc_id"), "citation source_doc mismatch")
            assert_true(citation.get("chunk_id") == evidence.get("chunk_id"), "citation chunk mismatch")
            if citation.get("span") is not None:
                assert_true(isinstance(citation.get("span"), dict), "citation span must be an object")

            source_doc_id = citation["source_doc_id"]
            if source_doc_id not in checked_source_docs:
                source_get = http_json("GET", f"{api_base}/sources/{source_doc_id}/presign-get", None, headers)
                assert_true(http_get_status(source_get["url"]) == 200, "presigned source GET should return 200")
                checked_source_docs.add(source_doc_id)

    artifacts = http_json("GET", f"{api_base}/runs/{run_id}/artifacts", None, headers)
    feedback = http_json("GET", f"{api_base}/runs/{run_id}/feedback", None, headers)
    assert_true(isinstance(artifacts, list), "artifacts endpoint should be compatible")
    assert_true(isinstance(feedback, list), "feedback endpoint should be compatible")
    artifact_count_before = len(artifacts)

    # idempotency: same input rerun should not create duplicated artifacts
    http_json("POST", f"{api_base}/runs/{run_id}/pipeline", {}, headers)
    time.sleep(2)
    artifacts_after = http_json("GET", f"{api_base}/runs/{run_id}/artifacts", None, headers)
    assert_true(
        len(artifacts_after) == artifact_count_before,
        "rerun with same input should not increase artifact count",
    )

    run_row = http_json("GET", f"{api_base}/runs/{run_id}", None, headers)
    assert_true(
        run_row.get("status") in {"success", "success_partial"},
        "run status should be success or success_partial in normal flow",
    )
    return run_id, len(issues)


def test_failure_stage_record(api_base: str, headers: dict) -> str:
    run = http_json(
        "POST",
        f"{api_base}/runs",
        {"task_prompt": "acceptance failure", "metadata": {"from": "script"}},
        headers,
    )
    run_id = run["id"]
    http_json("POST", f"{api_base}/runs/{run_id}/pipeline", {}, headers)

    def wait_failed_stage():
        rows = http_json("GET", f"{api_base}/runs/{run_id}/stages", None, headers)
        if not isinstance(rows, list):
            return None
        for row in rows:
            if row.get("stage_name") == "generate_feedback_with_citations":
                return rows
        return None

    stages = poll(wait_failed_stage, timeout_sec=40)
    failed = [
        s
        for s in stages
        if s.get("stage_name") == "generate_feedback_with_citations" and s.get("status") == "failed"
    ]
    assert_true(len(failed) == 1, "failed stage should be persisted for missing source case")
    assert_true(bool((failed[0].get("output_ref") or {}).get("error")), "failed stage should contain error message")
    assert_true(
        failed[0].get("failure_type") == "evidence_insufficient",
        "failed stage should be classified as evidence_insufficient",
    )
    assert_true(
        bool(failed[0].get("failure_detail")),
        "failed stage should include failure_detail",
    )
    run_row = http_json("GET", f"{api_base}/runs/{run_id}", None, headers)
    assert_true(
        run_row.get("status") == "blocked_evidence",
        "run status should become blocked_evidence when evidence cannot be attached",
    )
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Acceptance smoke checks for diffUI")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--auth-user", default=None)
    parser.add_argument("--auth-password", default=None)
    args = parser.parse_args()

    try:
        headers = auth_headers(args.api_base, args.auth_user, args.auth_password)
        run_id, issue_count = test_normal_and_compat(args.api_base, headers)
        failure_run_id = test_failure_stage_record(args.api_base, headers)
    except (AssertionError, TimeoutError, urllib.error.URLError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("[OK] issues normal flow")
    print(f"  run_id={run_id} issues={issue_count}")
    print("[OK] compatibility endpoints")
    print("[OK] failure stage persistence")
    print(f"  failure_run_id={failure_run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
