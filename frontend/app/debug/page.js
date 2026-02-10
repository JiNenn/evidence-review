"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const DEFAULT_AUTH_USER = process.env.NEXT_PUBLIC_AUTH_DEV_USER || "admin";
const DEFAULT_AUTH_PASSWORD = process.env.NEXT_PUBLIC_AUTH_DEV_PASSWORD || "admin";

export default function DebugPage() {
  const [runId, setRunId] = useState("");
  const [runRow, setRunRow] = useState(null);
  const [stages, setStages] = useState([]);
  const [auditIssues, setAuditIssues] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [authUser, setAuthUser] = useState(DEFAULT_AUTH_USER);
  const [authPassword, setAuthPassword] = useState(DEFAULT_AUTH_PASSWORD);
  const [accessToken, setAccessToken] = useState("");

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
      throw new Error(`API ${response.status}: ${text}`);
    }
    return response.json();
  };

  const login = async () => {
    setBusy(true);
    setError("");
    try {
      const token = await callApi("/auth/token", {
        method: "POST",
        body: JSON.stringify({ username: authUser, password: authPassword }),
        headers: {}
      });
      setAccessToken(token.access_token);
    } catch (e) {
      setError(String(e.message || e));
      setAccessToken("");
    } finally {
      setBusy(false);
    }
  };

  const loadRunDebug = async () => {
    if (!runId) {
      setError("run_id を入力してください。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const [run, stageRows, issueRows] = await Promise.all([
        callApi(`/runs/${runId}`),
        callApi(`/runs/${runId}/stages`),
        callApi(`/runs/${runId}/audit/issues`)
      ]);
      setRunRow(run);
      setStages(stageRows);
      setAuditIssues(issueRows);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const rerunPipeline = async () => {
    if (!runId) {
      setError("run_id を入力してください。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await callApi(`/runs/${runId}/pipeline`, {
        method: "POST",
        body: JSON.stringify({})
      });
      await new Promise((resolve) => setTimeout(resolve, 1500));
      await loadRunDebug();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main>
      <h1>diffUI / Debug</h1>
      <p className="lead">Run単位の監査・復旧操作を行う画面。主画面には再実行ボタンを置かない方針。</p>
      <p className="mono">
        <a href="/">主画面へ戻る</a>
      </p>

      <section className="panel grid">
        <h2>Auth</h2>
        <div className="row">
          <input value={authUser} onChange={(e) => setAuthUser(e.target.value)} placeholder="username" />
          <input
            value={authPassword}
            onChange={(e) => setAuthPassword(e.target.value)}
            placeholder="password"
            type="password"
          />
          <button className="secondary" onClick={login} disabled={busy}>
            Token取得
          </button>
        </div>
        <p className="mono">token: {accessToken ? "set" : "unset"}</p>
      </section>

      <section className="panel grid" style={{ marginTop: 16 }}>
        <h2>Run Debug</h2>
        <div className="row">
          <input value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="run_id" />
        </div>
        <div className="row">
          <button className="secondary" onClick={loadRunDebug} disabled={busy || !runId}>
            状態読込
          </button>
          <button onClick={rerunPipeline} disabled={busy || !runId}>
            Run単位で再実行
          </button>
        </div>
        {error ? <p className="mono" style={{ color: "#b12704" }}>{error}</p> : null}
        {runRow ? (
          <p className="mono">
            status={runRow.status} / created_at={runRow.created_at}
          </p>
        ) : null}
      </section>

      <section className="grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Audit Issues</h2>
          <p className="mono">count={auditIssues.length}</p>
          {auditIssues.length === 0 ? (
            <p className="mono">監査対象Issueはありません。</p>
          ) : (
            auditIssues.map((issue) => (
              <div key={issue.id} className="panel" style={{ marginTop: 8 }}>
                <p>
                  <strong>{issue.title}</strong>
                </p>
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
                {stage.failure_detail ? (
                  <pre>{JSON.stringify(stage.failure_detail, null, 2)}</pre>
                ) : null}
              </div>
            ))
          )}
        </article>
      </section>
    </main>
  );
}
