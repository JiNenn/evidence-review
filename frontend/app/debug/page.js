"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { getStoredRoles, getStoredToken } from "../lib/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function sanitizeReturnTo(path) {
  if (!path || typeof path !== "string") return "/";
  if (!path.startsWith("/")) return "/";
  if (path.startsWith("//")) return "/";
  return path;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function DebugPage() {
  const router = useRouter();

  const [runId, setRunId] = useState("");
  const [returnTo, setReturnTo] = useState("/");
  const [runRow, setRunRow] = useState(null);
  const [runMetrics, setRunMetrics] = useState(null);
  const [stages, setStages] = useState([]);
  const [stageAttempts, setStageAttempts] = useState([]);
  const [auditIssues, setAuditIssues] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [accessToken, setAccessToken] = useState("");
  const [tokenRoles, setTokenRoles] = useState([]);
  const [authHydrated, setAuthHydrated] = useState(false);
  const [returnCountdown, setReturnCountdown] = useState(null);
  const [recoveryState, setRecoveryState] = useState("");

  const redirectToLogin = () => {
    const next = encodeURIComponent(
      typeof window !== "undefined" ? `${window.location.pathname}${window.location.search}` : "/debug"
    );
    router.replace(`/login?next=${next}`);
  };

  const callApi = async (path, options = {}) => {
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {})
    };
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers
    });
    if (!response.ok) {
      const text = await response.text();
      const err = new Error(`API ${response.status}: ${text}`);
      err.unauthorized = response.status === 401;
      err.forbidden = response.status === 403;
      throw err;
    }
    return response.json();
  };

  const handleApiError = (e) => {
    if (e?.unauthorized) {
      redirectToLogin();
      return;
    }
    if (e?.forbidden) {
      setError("権限不足です。Debug の監査機能は `audit/admin` ロールが必要です。");
      return;
    }
    setError(String(e.message || e));
  };

  const loadRunDebug = async (targetRunId = runId) => {
    if (!targetRunId) {
      setError("run_id を入力してください。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const [run, metrics, stageRows, attemptRows, issueRows] = await Promise.all([
        callApi(`/runs/${targetRunId}`),
        callApi(`/runs/${targetRunId}/metrics`),
        callApi(`/runs/${targetRunId}/stages`),
        callApi(`/runs/${targetRunId}/stage-attempts`),
        callApi(`/runs/${targetRunId}/audit/issues`)
      ]);
      setRunRow(run);
      setRunMetrics(metrics);
      setStages(stageRows);
      setStageAttempts(attemptRows);
      setAuditIssues(issueRows);
    } catch (e) {
      handleApiError(e);
    } finally {
      setBusy(false);
    }
  };

  const monitorRecovery = async (targetRunId) => {
    setRecoveryState("watching");
    for (let i = 0; i < 60; i += 1) {
      try {
        const run = await callApi(`/runs/${targetRunId}`);
        setRunRow(run);
        if (run.status === "success" || run.status === "success_partial") {
          setRecoveryState("recovered");
          setReturnCountdown(3);
          await loadRunDebug(targetRunId);
          return;
        }
        if (run.status === "blocked_evidence" || run.status === "failed_system" || run.status === "failed_legacy") {
          setRecoveryState("failed");
          await loadRunDebug(targetRunId);
          return;
        }
      } catch (e) {
        handleApiError(e);
        return;
      }
      await sleep(2000);
    }
    setRecoveryState("timeout");
  };

  const rerunPipeline = async () => {
    if (!runId) {
      setError("run_id を入力してください。");
      return;
    }
    setBusy(true);
    setError("");
    setReturnCountdown(null);
    try {
      await callApi(`/runs/${runId}/pipeline`, {
        method: "POST",
        body: JSON.stringify({})
      });
      await monitorRecovery(runId);
    } catch (e) {
      handleApiError(e);
    } finally {
      setBusy(false);
    }
  };

  const latestFailedStage = stages.find((row) => row.status === "failed") || null;
  const failureSummary = latestFailedStage?.failure_detail?.summary || latestFailedStage?.output_ref?.error || "";

  useEffect(() => {
    setAccessToken(getStoredToken());
    setTokenRoles(getStoredRoles());
    const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
    const runParam = params.get("run_id") || "";
    const returnParam = sanitizeReturnTo(params.get("return_to") || "/");
    setRunId(runParam);
    setReturnTo(returnParam);
    setAuthHydrated(true);
  }, []);

  useEffect(() => {
    if (!authHydrated) {
      return;
    }
    if (accessToken) {
      return;
    }
    redirectToLogin();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, authHydrated]);

  useEffect(() => {
    if (!authHydrated || !runId || !accessToken) {
      return;
    }
    loadRunDebug(runId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, accessToken, authHydrated]);

  useEffect(() => {
    if (returnCountdown == null) return undefined;
    if (returnCountdown <= 0) {
      router.replace(returnTo);
      return undefined;
    }
    const timer = setTimeout(() => setReturnCountdown((prev) => (prev == null ? null : prev - 1)), 1000);
    return () => clearTimeout(timer);
  }, [returnCountdown, returnTo, router]);

  return (
    <main>
      <h1>diffUI / Debug</h1>
      <p className="lead">Run単位の監査・復旧タスク画面（共通セッション前提）。</p>
      <p className="mono">
        <a href="/">Homeへ戻る</a> / return_to={returnTo}
      </p>

      <section className="panel grid">
        <h2>Run Debug</h2>
        <p className="mono">roles: {tokenRoles.join(",") || "-"}</p>
        <div className="row">
          <input value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="run_id" />
        </div>
        <div className="row">
          <button className="secondary" onClick={() => loadRunDebug()} disabled={busy || !runId}>
            状態読込
          </button>
          <button onClick={rerunPipeline} disabled={busy || !runId}>
            Run単位で再実行
          </button>
          <button className="secondary" onClick={() => router.replace(returnTo)}>
            元画面へ戻る
          </button>
        </div>
        {error ? <p className="mono status-ng">{error}</p> : null}
        {runRow ? (
          <p className="mono">
            status={runRow.status} / created_at={runRow.created_at} / recovery_state={recoveryState || "-"}
          </p>
        ) : null}
      </section>

      {returnCountdown != null ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <p className="mono">復旧完了。{returnCountdown}秒後に元画面へ戻ります。</p>
          <div className="row">
            <button className="secondary" onClick={() => router.replace(returnTo)}>
              今すぐ戻る
            </button>
            <button className="secondary" onClick={() => setReturnCountdown(null)}>
              このままDebugに残る
            </button>
          </div>
        </section>
      ) : null}

      {recoveryState === "failed" ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>Recovery Failed</h2>
          <p className="mono">summary: {failureSummary || "-"}</p>
          <p className="mono">次アクション: 1) source追加 2) retry 3) stage detail確認</p>
        </section>
      ) : null}

      <section className="grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Audit Issues</h2>
          <p className="mono">count={auditIssues.length}</p>
          {auditIssues.length === 0 ? (
            <p className="mono">監査対象Issueはありません。</p>
          ) : (
            auditIssues.map((issue) => (
              <div key={issue.id} className="panel" style={{ marginTop: 8 }}>
                <p><strong>{issue.title}</strong></p>
                <p>{issue.summary}</p>
                <p className="mono">
                  severity={issue.severity} / status={issue.status} / evidences={issue.evidence_count}
                </p>
              </div>
            ))
          )}
        </article>

        <article className="panel">
          <h2>Stages</h2>
          {stages.length === 0 ? (
            <p className="mono">stage情報はまだありません。</p>
          ) : (
            stages.map((stage) => (
              <div key={`${stage.stage_name}-${stage.attempt}`} className="panel" style={{ marginTop: 8 }}>
                <p className="mono">
                  {stage.stage_name} / status={stage.status} / attempt={stage.attempt}
                </p>
                {stage.failure_type ? <p className="mono">failure_type: {stage.failure_type}</p> : null}
                {stage.failure_detail ? <pre>{JSON.stringify(stage.failure_detail, null, 2)}</pre> : null}
              </div>
            ))
          )}
        </article>

        <article className="panel">
          <h2>Stage Attempts</h2>
          <p className="mono">count={stageAttempts.length}</p>
          {stageAttempts.length === 0 ? (
            <p className="mono">attempt履歴はまだありません。</p>
          ) : (
            stageAttempts.map((row) => (
              <div key={row.id} className="panel" style={{ marginTop: 8 }}>
                <p className="mono">
                  {row.stage_name} / attempt_no={row.attempt_no} / status={row.status}
                </p>
                {row.failure_type ? <p className="mono">failure_type: {row.failure_type}</p> : null}
                {row.failure_detail ? <pre>{JSON.stringify(row.failure_detail, null, 2)}</pre> : null}
              </div>
            ))
          )}
        </article>

        <article className="panel">
          <h2>Run Metrics</h2>
          {!runMetrics ? (
            <p className="mono">metrics はまだありません。</p>
          ) : (
            <>
              <p className="mono">
                issues={runMetrics.issue_total} / visible={runMetrics.visible_issue_count} / hidden=
                {runMetrics.hidden_issue_count}
              </p>
              <p className="mono">
                selection_scores={runMetrics.selection_score.count} / avg={runMetrics.selection_score.avg ?? "-"} / p90=
                {runMetrics.selection_score.p90 ?? "-"}
              </p>
              <p className="mono">
                retried_stages={runMetrics.retry.retried_stage_count} / retry_success=
                {runMetrics.retry.retry_success_count} / retry_failure={runMetrics.retry.retry_failure_count}
              </p>
            </>
          )}
        </article>
      </section>
    </main>
  );
}
