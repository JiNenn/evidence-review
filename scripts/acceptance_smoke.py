#!/usr/bin/env python3
import argparse
import http.client
import json
import socket
import sys
import time
import urllib.error
import urllib.request

NORMAL_TASK_PROMPT = "acceptance normal"
NORMAL_SOURCE_FILENAME = "acceptance.txt"
NORMAL_SOURCE_BODY = "この文書はレビュー論点抽出の根拠です。\n改善対象の差分を示します。"

TRANSIENT_HTTP_ERRORS = (
    urllib.error.URLError,
    http.client.RemoteDisconnected,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
)


def urlopen_with_retry(
    req: urllib.request.Request,
    *,
    timeout: int = 20,
    max_attempts: int = 5,
    initial_backoff_sec: float = 0.4,
):
    backoff = initial_backoff_sec
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except TRANSIENT_HTTP_ERRORS as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 4.0)
    raise last_exc  # type: ignore[misc]


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
    with urlopen_with_retry(req, timeout=20) as resp:
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
    with urlopen_with_retry(req, timeout=20):
        return


def http_get_status(url: str) -> int:
    req = urllib.request.Request(url, method="GET")
    with urlopen_with_retry(req, timeout=20) as resp:
        _ = resp.read(1)
        return int(resp.status)


def http_status_or_error(
    method: str,
    url: str,
    payload: dict | None = None,
    extra_headers: dict | None = None,
) -> tuple[int, dict | list | str]:
    body = None
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen_with_retry(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return int(resp.status), {}
            return int(resp.status), json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return int(exc.code), raw


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


def as_float(value, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def auth_headers(api_base: str, auth_user: str | None, auth_password: str | None) -> dict:
    if not auth_user:
        return {}
    token = http_json(
        "POST",
        f"{api_base}/auth/token",
        {"username": auth_user, "password": auth_password or ""},
    )
    return {"Authorization": f"Bearer {token['access_token']}"}


def create_run_with_source_and_pipeline(
    api_base: str,
    headers: dict,
    *,
    task_prompt: str,
    filename: str,
    source_body: str,
) -> str:
    run = http_json(
        "POST",
        f"{api_base}/runs",
        {"task_prompt": task_prompt, "metadata": {"from": "script", "kind": "repro"}},
        headers,
    )
    run_id = run["id"]

    presign = http_json(
        "POST",
        f"{api_base}/runs/{run_id}/sources/presign-put",
        {"filename": filename, "content_type": "text/plain"},
        headers,
    )
    http_put_bytes(
        presign["url"],
        source_body.encode("utf-8"),
        "text/plain",
    )
    http_json(
        "POST",
        f"{api_base}/runs/{run_id}/sources/ingest",
        {
            "source_doc_id": presign["source_doc_id"],
            "title": filename,
            "content_type": "text/plain",
        },
        headers,
    )
    time.sleep(2)
    http_json("POST", f"{api_base}/runs/{run_id}/pipeline", {}, headers)
    wait_for_issues_ready(api_base, headers, run_id, timeout_sec=180)
    return run_id


def extract_selection_signature(issue_detail: dict) -> dict:
    evidences = issue_detail.get("evidences", [])
    assert_true(len(evidences) >= 1, "issue detail should include evidences for signature")

    def score_tuple(evidence: dict):
        selection = evidence.get("selection")
        if not isinstance(selection, dict):
            citation_span = evidence.get("citation_span") or {}
            selection = citation_span.get("selection") if isinstance(citation_span, dict) else {}
        if not isinstance(selection, dict):
            selection = {}
        return (
            as_float(selection.get("combined_score"), -1.0),
            as_float(selection.get("search_score"), -1.0),
            as_float(selection.get("lexical_score"), -1.0),
            -as_float(selection.get("search_rank"), 10**9),
            evidence.get("id", ""),
        )

    best = sorted(evidences, key=score_tuple, reverse=True)[0]
    selection = best.get("selection")
    if not isinstance(selection, dict):
        citation_span = best.get("citation_span") or {}
        selection = citation_span.get("selection") if isinstance(citation_span, dict) else {}
    if not isinstance(selection, dict):
        selection = {}

    return {
        "chunk_text": best.get("chunk_text", ""),
        "combined_score": round(as_float(selection.get("combined_score"), -1.0), 6),
        "search_score": round(as_float(selection.get("search_score"), -1.0), 6),
        "lexical_score": round(as_float(selection.get("lexical_score"), -1.0), 6),
        "selection_version": selection.get("version"),
    }


def collect_run_selection_signature(api_base: str, headers: dict, run_id: str) -> list[dict]:
    issues = http_json("GET", f"{api_base}/runs/{run_id}/issues", None, headers)
    assert_true(isinstance(issues, list), "issues endpoint should return list")
    signature_rows = []
    sorted_issues = sorted(
        issues,
        key=lambda row: (row.get("title", ""), row.get("summary", ""), int(row.get("severity", 0))),
    )
    for issue in sorted_issues:
        detail = http_json("GET", f"{api_base}/issues/{issue['id']}", None, headers)
        signature_rows.append(
            {
                "title": issue.get("title", ""),
                "summary": issue.get("summary", ""),
                "severity": issue.get("severity", 0),
                **extract_selection_signature(detail),
            }
        )
    return signature_rows


def selection_projection(signature_rows: list[dict]) -> list[tuple]:
    """
    Compare only evidence-selection characteristics so LLM wording variance
    in issue titles/summaries does not break reproducibility checks.
    """
    projected = {
        (
            row.get("selection_version"),
            row.get("chunk_text", ""),
        )
        for row in signature_rows
    }
    return sorted(projected)


def wait_search_success(api_base: str, headers: dict, search_id: str) -> dict:
    def poll_search():
        row = http_json("GET", f"{api_base}/search/{search_id}", None, headers)
        status = row.get("status")
        if status == "success":
            return row
        if status == "failed":
            raise AssertionError("search should not fail in vector smoke")
        return None

    return poll(poll_search, timeout_sec=40)


def wait_for_issues_ready(api_base: str, headers: dict, run_id: str, timeout_sec: int = 180) -> list[dict]:
    started = time.time()
    while True:
        issues = http_json("GET", f"{api_base}/runs/{run_id}/issues", None, headers)
        if isinstance(issues, list) and len(issues) > 0:
            return issues

        run_row = http_json("GET", f"{api_base}/runs/{run_id}", None, headers)
        run_status = str(run_row.get("status") or "")
        if run_status in {"failed_system", "blocked_evidence", "failed_legacy"}:
            stages = http_json("GET", f"{api_base}/runs/{run_id}/stages", None, headers)
            failed_rows = [row for row in stages if row.get("status") == "failed"] if isinstance(stages, list) else []
            latest_failed = failed_rows[-1] if failed_rows else {}
            detail = latest_failed.get("failure_detail") or {}
            summary = detail.get("summary") or latest_failed.get("output_ref", {}).get("error") or "unknown"
            raise AssertionError(f"run ended in {run_status}: {summary}")

        if time.time() - started > timeout_sec:
            raise TimeoutError(f"poll timeout (run_status={run_status})")
        time.sleep(1.0)


def test_normal_and_compat(api_base: str, headers: dict) -> tuple[str, int, str]:
    public_runs_before = http_json("GET", f"{api_base}/runs")
    assert_true(isinstance(public_runs_before, list), "public runs endpoint should return list")

    run = http_json(
        "POST",
        f"{api_base}/runs",
        {"task_prompt": NORMAL_TASK_PROMPT, "metadata": {"from": "script"}},
        headers,
    )
    run_id = run["id"]
    public_runs_after = http_json("GET", f"{api_base}/runs")
    run_row = next((row for row in public_runs_after if row.get("id") == run_id), None)
    assert_true(bool(run_row), "public runs should include newly created run")
    assert_true("title" in run_row, "public run row should include title")
    assert_true("source_count" in run_row, "public run row should include source_count")
    assert_true("issue_count" in run_row, "public run row should include issue_count")
    assert_true("visible_issue_count" in run_row, "public run row should include visible_issue_count")

    presign = http_json(
        "POST",
        f"{api_base}/runs/{run_id}/sources/presign-put",
        {"filename": NORMAL_SOURCE_FILENAME, "content_type": "text/plain"},
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
        NORMAL_SOURCE_BODY.encode("utf-8"),
        "text/plain",
    )

    http_json(
        "POST",
        f"{api_base}/runs/{run_id}/sources/ingest",
        {
            "source_doc_id": presign["source_doc_id"],
            "title": NORMAL_SOURCE_FILENAME,
            "content_type": "text/plain",
        },
        headers,
    )
    time.sleep(2)
    http_json("POST", f"{api_base}/runs/{run_id}/pipeline", {}, headers)

    issues = wait_for_issues_ready(api_base, headers, run_id, timeout_sec=180)
    assert_true(isinstance(issues, list) and len(issues) > 0, "issues should exist")
    assert_true(all(issue.get("evidence_count", 0) >= 1 for issue in issues), "all issues must have evidence")
    assert_true(all(issue.get("status") != "hidden" for issue in issues), "hidden issues must not be listed")
    stage_rows = http_json("GET", f"{api_base}/runs/{run_id}/stages", None, headers)
    derive_stage = next(
        (
            row
            for row in stage_rows
            if row.get("stage_name") == "derive_issues_from_changes" and row.get("status") == "success"
        ),
        None,
    )
    assert_true(bool(derive_stage), "derive_issues_from_changes stage should succeed")
    derive_output = (derive_stage or {}).get("output_ref") or {}
    assert_true("dedup_similarity_threshold" in derive_output, "derive stage should expose dedup threshold")
    assert_true("dedup_merged_count" in derive_output, "derive stage should expose dedup merged count")
    stage_attempt_rows = http_json("GET", f"{api_base}/runs/{run_id}/stage-attempts", None, headers)
    assert_true(isinstance(stage_attempt_rows, list) and len(stage_attempt_rows) > 0, "stage attempts should exist")
    assert_true(
        any(
            row.get("stage_name") == "derive_issues_from_changes"
            and row.get("attempt_no", 0) >= 1
            and row.get("status") == "success"
            for row in stage_attempt_rows
        ),
        "stage attempts should include successful derive_issues_from_changes",
    )
    metrics = http_json("GET", f"{api_base}/runs/{run_id}/metrics", None, headers)
    assert_true(metrics.get("run_id") == run_id, "run metrics should include matching run_id")
    assert_true("hidden_rate" in metrics, "run metrics should include hidden_rate")
    assert_true("selection_score" in metrics, "run metrics should include selection_score")
    assert_true("retry" in metrics, "run metrics should include retry summary")

    issues_all = http_json("GET", f"{api_base}/runs/{run_id}/issues?include_hidden=true", None, headers)
    assert_true(isinstance(issues_all, list), "issues include_hidden should return list")
    assert_true(len(issues_all) >= len(issues), "include_hidden should return at least as many issues")
    issues_audit = http_json("GET", f"{api_base}/runs/{run_id}/audit/issues", None, headers)
    assert_true(isinstance(issues_audit, list), "audit issues endpoint should return list")
    assert_true(len(issues_audit) >= len(issues), "audit issues should return at least as many issues")

    checked_source_docs: set[str] = set()
    for issue in issues:
        issue_detail = http_json("GET", f"{api_base}/issues/{issue['id']}", None, headers)
        evidences = issue_detail.get("evidences", [])
        assert_true(len(evidences) >= 1, "issue detail should include evidences")

        for evidence in evidences:
            assert_true(bool(evidence.get("chunk_loc")), "evidence chunk_loc must exist")
            assert_true(bool(evidence.get("source_doc_id")), "evidence source_doc_id must exist")
            assert_true(bool(evidence.get("chunk_id")), "evidence chunk_id must exist")
            if evidence.get("loc") is not None:
                assert_true(isinstance(evidence.get("loc"), dict), "evidence loc must be an object")
            if evidence.get("citation_span") is not None:
                assert_true(isinstance(evidence.get("citation_span"), dict), "citation_span must be an object")

            citation = http_json("GET", f"{api_base}/citations/{evidence['citation_id']}", None, headers)
            assert_true(citation.get("source_doc_id") == evidence.get("source_doc_id"), "citation source_doc mismatch")
            assert_true(citation.get("chunk_id") == evidence.get("chunk_id"), "citation chunk mismatch")
            if citation.get("span") is not None:
                assert_true(isinstance(citation.get("span"), dict), "citation span must be an object")
            if citation.get("feedback_id") is None:
                span = citation.get("span") or {}
                selection = span.get("selection")
                assert_true(bool(evidence.get("selection")), "issue evidence should include explicit selection")
                assert_true(bool(selection), "issue citation span should include selection detail")
                assert_true(
                    selection.get("version") == "search_score_weighted_v1",
                    "issue citation selection version mismatch",
                )
                assert_true(
                    isinstance(selection.get("combined_score"), (int, float)),
                    "issue citation selection should include combined_score",
                )
                assert_true(
                    evidence.get("selection", {}).get("version") == selection.get("version"),
                    "issue evidence selection should match citation span selection",
                )

            source_doc_id = citation["source_doc_id"]
            if source_doc_id not in checked_source_docs:
                source_get = http_json("GET", f"{api_base}/sources/{source_doc_id}/presign-get", None, headers)
                assert_true(http_get_status(source_get["url"]) == 200, "presigned source GET should return 200")
                checked_source_docs.add(source_doc_id)

    artifacts = http_json("GET", f"{api_base}/runs/{run_id}/artifacts", None, headers)
    feedback = http_json("GET", f"{api_base}/runs/{run_id}/feedback", None, headers)
    assert_true(isinstance(artifacts, list), "artifacts endpoint should be compatible")
    assert_true(isinstance(feedback, list), "feedback endpoint should be compatible")
    improved_artifacts = [row for row in artifacts if row.get("kind") == "improved"]
    assert_true(len(improved_artifacts) >= 1, "improved artifact should be generated")
    latest_improved = sorted(improved_artifacts, key=lambda row: row.get("created_at", ""))[-1]
    improved_text = (latest_improved.get("content_text") or "").lstrip()
    assert_true(
        improved_text.startswith("#"),
        "improved artifact should start with markdown heading",
    )
    artifact_count_before = len(artifacts)

    vector_search = http_json(
        "POST",
        f"{api_base}/runs/{run_id}/search",
        {"query": "レビュー論点の根拠", "mode": "vector", "filters": {"top_k": 5, "min_score": 0.0}},
        headers,
    )
    vector_search_full = wait_search_success(api_base, headers, vector_search["id"])
    vector_results = vector_search_full.get("results", [])
    assert_true(isinstance(vector_results, list) and len(vector_results) > 0, "vector search should return results")
    assert_true(
        all((row.get("payload") or {}).get("score_reason") == "char_ngram_cosine" for row in vector_results),
        "vector search results should include score_reason=char_ngram_cosine",
    )

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

    # reproducibility:
    # 1) same run + idempotent rerun must preserve full selection signatures
    # 2) fresh run with same input must preserve projected selection characteristics
    baseline_signature = collect_run_selection_signature(api_base, headers, run_id)
    http_json("POST", f"{api_base}/runs/{run_id}/pipeline", {}, headers)
    wait_for_issues_ready(api_base, headers, run_id, timeout_sec=180)
    rerun_signature = collect_run_selection_signature(api_base, headers, run_id)
    assert_true(
        rerun_signature == baseline_signature,
        "selection signature should be reproducible on idempotent rerun",
    )

    repro_run_id = create_run_with_source_and_pipeline(
        api_base,
        headers,
        task_prompt=NORMAL_TASK_PROMPT,
        filename=NORMAL_SOURCE_FILENAME,
        source_body=NORMAL_SOURCE_BODY,
    )
    repro_signature = collect_run_selection_signature(api_base, headers, repro_run_id)
    assert_true(
        selection_projection(repro_signature) == selection_projection(baseline_signature),
        "selection projection should be reproducible for the same input",
    )
    return run_id, len(issues), repro_run_id


def test_audit_access_control(
    api_base: str,
    run_id: str,
    viewer_user: str | None,
    viewer_password: str | None,
) -> None:
    if not viewer_user:
        return
    try:
        viewer_headers = auth_headers(api_base, viewer_user, viewer_password)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return
        raise
    status_code, _ = http_status_or_error(
        "GET",
        f"{api_base}/runs/{run_id}/issues?include_hidden=true",
        None,
        viewer_headers,
    )
    assert_true(status_code == 403, "viewer should not access include_hidden issues")
    status_code, _ = http_status_or_error(
        "GET",
        f"{api_base}/runs/{run_id}/audit/issues",
        None,
        viewer_headers,
    )
    assert_true(status_code == 403, "viewer should not access audit issues endpoint")


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
    failure_detail = failed[0].get("failure_detail") or {}
    assert_true(
        failure_detail.get("required") == "source_chunk",
        "failure detail should include required input for blocked_evidence",
    )
    assert_true(
        bool(failure_detail.get("summary")),
        "failure detail should include summary for operator readability",
    )
    run_row = http_json("GET", f"{api_base}/runs/{run_id}", None, headers)
    assert_true(
        run_row.get("status") == "blocked_evidence",
        "run status should become blocked_evidence when evidence cannot be attached",
    )
    failed_attempts = http_json(
        "GET",
        f"{api_base}/runs/{run_id}/stage-attempts?stage_name=generate_feedback_with_citations",
        None,
        headers,
    )
    assert_true(
        any(
            row.get("stage_name") == "generate_feedback_with_citations"
            and row.get("status") == "failed"
            and row.get("failure_type") == "evidence_insufficient"
            for row in failed_attempts
        ),
        "failed stage attempt should be persisted with evidence_insufficient",
    )
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Acceptance smoke checks for diffUI")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--auth-user", default=None)
    parser.add_argument("--auth-password", default=None)
    parser.add_argument("--viewer-user", default="viewer")
    parser.add_argument("--viewer-password", default="viewer")
    args = parser.parse_args()

    try:
        headers = auth_headers(args.api_base, args.auth_user, args.auth_password)
        run_id, issue_count, repro_run_id = test_normal_and_compat(args.api_base, headers)
        if args.auth_user:
            test_audit_access_control(args.api_base, run_id, args.viewer_user, args.viewer_password)
        failure_run_id = test_failure_stage_record(args.api_base, headers)
    except (AssertionError, TimeoutError, urllib.error.URLError, http.client.HTTPException) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("[OK] issues normal flow")
    print(f"  run_id={run_id} issues={issue_count}")
    print("[OK] compatibility endpoints")
    print("[OK] selection reproducibility")
    print(f"  repro_run_id={repro_run_id}")
    print("[OK] failure stage persistence")
    print(f"  failure_run_id={failure_run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
