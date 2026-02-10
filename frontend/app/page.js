"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const DEFAULT_AUTH_USER = process.env.NEXT_PUBLIC_AUTH_DEV_USER || "admin";
const DEFAULT_AUTH_PASSWORD = process.env.NEXT_PUBLIC_AUTH_DEV_PASSWORD || "admin";

export default function HomePage() {
  const [taskPrompt, setTaskPrompt] = useState("根拠付きでレビュー論点を抽出してください。");
  const [runId, setRunId] = useState("");
  const [issues, setIssues] = useState([]);
  const [issueDetails, setIssueDetails] = useState({});
  const [loadingIssueId, setLoadingIssueId] = useState("");
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

  const refreshIssues = async (id) => {
    const rows = await callApi(`/runs/${id}/issues`);
    setIssues(rows);
  };

  const createRun = async () => {
    setBusy(true);
    setError("");
    try {
      const run = await callApi("/runs", {
        method: "POST",
        body: JSON.stringify({ task_prompt: taskPrompt, metadata: { created_from: "frontend" } })
      });
      setRunId(run.id);
      setIssues([]);
      setIssueDetails({});
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const runPipeline = async () => {
    if (!runId) {
      setError("先に Run を作成してください。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await callApi(`/runs/${runId}/pipeline`, {
        method: "POST",
        body: JSON.stringify({})
      });
      await refreshIssues(runId);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const reload = async () => {
    if (!runId) {
      return;
    }
    setBusy(true);
    setError("");
    try {
      await refreshIssues(runId);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const openIssueDetail = async (issueId) => {
    setLoadingIssueId(issueId);
    setError("");
    try {
      const detail = await callApi(`/issues/${issueId}`);
      setIssueDetails((prev) => ({ ...prev, [issueId]: detail }));
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoadingIssueId("");
    }
  };

  return (
    <main>
      <h1>diffUI / Issues-First Review</h1>
      <p className="lead">差分そのものではなく、根拠付きの論点を中心にレビューする画面。</p>

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
        <label>
          Task Prompt
          <textarea rows={4} value={taskPrompt} onChange={(e) => setTaskPrompt(e.target.value)} />
        </label>
        <div className="row">
          <button onClick={createRun} disabled={busy}>
            Run 作成
          </button>
          <button className="secondary" onClick={runPipeline} disabled={busy || !runId}>
            Pipeline 実行
          </button>
          <button className="secondary" onClick={reload} disabled={busy || !runId}>
            再読込
          </button>
        </div>
        <div className="mono">
          <span className="chip">API: {API_BASE}</span>
          <span className="chip">run_id: {runId || "-"}</span>
          <span className="chip">issues: {issues.length}</span>
        </div>
        {error ? <p className="mono" style={{ color: "#b12704" }}>{error}</p> : null}
      </section>

      <section className="grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Issues</h2>
          {issues.length === 0 ? (
            <p className="mono">表示可能な論点はまだありません。</p>
          ) : (
            issues.map((issue) => (
              <div key={issue.id} className="panel" style={{ marginTop: 10 }}>
                <p>
                  <strong>{issue.title}</strong>
                </p>
                <p>{issue.summary}</p>
                <p className="mono">
                  severity={issue.severity} / confidence={issue.confidence.toFixed(2)} / status={issue.status}
                </p>
                <p className="mono">evidences={issue.evidence_count}</p>
                <button
                  className="secondary"
                  onClick={() => openIssueDetail(issue.id)}
                  disabled={loadingIssueId === issue.id}
                >
                  詳細を表示
                </button>

                {issueDetails[issue.id] ? (
                  <div className="panel" style={{ marginTop: 10 }}>
                    {issueDetails[issue.id].evidences.length === 0 ? (
                      <p className="mono">この論点には表示可能な根拠がありません。</p>
                    ) : (
                      issueDetails[issue.id].evidences.map((ev) => (
                        <div key={ev.id} className="panel" style={{ marginTop: 8 }}>
                          <p className="mono">source: {ev.source_title}</p>
                          {ev.before_excerpt ? (
                            <>
                              <p className="mono">before</p>
                              <pre>{ev.before_excerpt}</pre>
                            </>
                          ) : null}
                          {ev.after_excerpt ? (
                            <>
                              <p className="mono">after</p>
                              <pre>{ev.after_excerpt}</pre>
                            </>
                          ) : null}
                          <p className="mono">evidence snippet</p>
                          <pre>{ev.chunk_text}</pre>
                          <a href={ev.source_presigned_url} target="_blank" rel="noreferrer" className="mono">
                            原文を開く
                          </a>
                        </div>
                      ))
                    )}
                  </div>
                ) : null}
              </div>
            ))
          )}
        </article>
      </section>
    </main>
  );
}
