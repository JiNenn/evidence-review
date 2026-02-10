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
  const [includeHidden, setIncludeHidden] = useState(false);
  const [stages, setStages] = useState([]);
  const [sourceDocs, setSourceDocs] = useState([]);
  const [sourceFile, setSourceFile] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [loadingIssueId, setLoadingIssueId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [authUser, setAuthUser] = useState(DEFAULT_AUTH_USER);
  const [authPassword, setAuthPassword] = useState(DEFAULT_AUTH_PASSWORD);
  const [accessToken, setAccessToken] = useState("");

  const formatScore = (value) => {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "-";
    }
    return value.toFixed(3);
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
    const query = includeHidden ? "?include_hidden=true" : "";
    const rows = await callApi(`/runs/${id}/issues${query}`);
    setIssues(rows);
  };

  const refreshStages = async (id) => {
    const rows = await callApi(`/runs/${id}/stages`);
    setStages(rows);
    return rows;
  };

  const waitPipeline = async (id) => {
    const maxPoll = 60;
    for (let i = 0; i < maxPoll; i += 1) {
      const rows = await refreshStages(id);
      const failed = rows.find((row) => row.status === "failed");
      if (failed) {
        const reason = failed.output_ref?.error || "unknown error";
        throw new Error(`stage failed: ${failed.stage_name} / ${reason}`);
      }
      const done = rows.find(
        (row) => row.stage_name === "attach_evidence_to_issues" && row.status === "success"
      );
      if (done) {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    throw new Error("pipeline timeout");
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
      setStages([]);
      setSourceDocs([]);
      setSourceFile(null);
      setFileInputKey((prev) => prev + 1);
      setIssueDetails({});
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const uploadSource = async () => {
    if (!runId) {
      setError("先に Run を作成してください。");
      return;
    }
    if (!sourceFile) {
      setError("アップロードするファイルを選択してください。");
      return;
    }

    setBusy(true);
    setError("");
    try {
      const contentType = sourceFile.type || "application/octet-stream";
      const presign = await callApi(`/runs/${runId}/sources/presign-put`, {
        method: "POST",
        body: JSON.stringify({
          filename: sourceFile.name,
          content_type: contentType
        })
      });

      const putResp = await fetch(presign.url, {
        method: "PUT",
        headers: {
          "Content-Type": contentType
        },
        body: sourceFile
      });
      if (!putResp.ok) {
        const text = await putResp.text();
        throw new Error(`upload failed: ${putResp.status} ${text}`);
      }

      const ingest = await callApi(`/runs/${runId}/sources/ingest`, {
        method: "POST",
        body: JSON.stringify({
          source_doc_id: presign.source_doc_id,
          object_key: presign.object_key,
          title: sourceFile.name,
          content_type: contentType
        })
      });

      setSourceDocs((prev) => [
        ...prev,
        {
          id: ingest.source_doc_id,
          title: sourceFile.name
        }
      ]);
      setSourceFile(null);
      setFileInputKey((prev) => prev + 1);
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
    if (sourceDocs.length < 1) {
      setError("先に少なくとも1件のソースを取り込んでください。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await callApi(`/runs/${runId}/pipeline`, {
        method: "POST",
        body: JSON.stringify({})
      });
      await waitPipeline(runId);
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
      await refreshStages(runId);
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
          <span className="chip">sources: {sourceDocs.length}</span>
          <span className="chip">issues: {issues.length}</span>
          <span className="chip">stages: {stages.length}</span>
        </div>
        <label className="mono">
          <input
            type="checkbox"
            checked={includeHidden}
            onChange={(e) => setIncludeHidden(e.target.checked)}
            style={{ marginRight: 6 }}
          />
          hidden issue を含めて取得
        </label>
        {error ? <p className="mono" style={{ color: "#b12704" }}>{error}</p> : null}
      </section>

      <section className="grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Sources</h2>
          <div className="row">
            <input
              key={fileInputKey}
              type="file"
              onChange={(e) => setSourceFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)}
            />
            <button className="secondary" onClick={uploadSource} disabled={busy || !runId || !sourceFile}>
              アップロードして取り込み
            </button>
          </div>
          {sourceDocs.length === 0 ? (
            <p className="mono">取り込み済みソースはまだありません。</p>
          ) : (
            sourceDocs.map((doc) => (
              <p className="mono" key={doc.id}>
                {doc.id} / {doc.title}
              </p>
            ))
          )}
        </article>

        <article className="panel">
          <h2>Stages</h2>
          {stages.length === 0 ? (
            <p className="mono">stage情報はまだありません。</p>
          ) : (
            stages.map((stage) => (
              <div key={stage.stage_name} className="panel" style={{ marginTop: 10 }}>
                <p className="mono">
                  {stage.stage_name} / status={stage.status} / attempt={stage.attempt}
                </p>
                {stage.failure_type ? <p className="mono">failure_type: {stage.failure_type}</p> : null}
                {stage.failure_detail ? (
                  <pre>{JSON.stringify(stage.failure_detail, null, 2)}</pre>
                ) : null}
                {stage.output_ref?.error ? <p className="mono">error: {stage.output_ref.error}</p> : null}
              </div>
            ))
          )}
        </article>

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
                      issueDetails[issue.id].evidences.map((ev) => {
                        const selection =
                          ev.citation_span && typeof ev.citation_span === "object" ? ev.citation_span.selection : null;
                        const selectionWeights =
                          selection && typeof selection === "object" ? selection.weights || null : null;
                        return (
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
                            {selection ? (
                              <div className="panel" style={{ marginTop: 8 }}>
                                <p className="mono">
                                  selection={selection.version || "unknown"} / combined=
                                  {formatScore(selection.combined_score)}
                                </p>
                                <p className="mono">
                                  search={formatScore(selection.search_score)} / lexical=
                                  {formatScore(selection.lexical_score)} / rank=
                                  {selection.search_rank ?? "-"}
                                </p>
                                <p className="mono">
                                  w(search)={formatScore(selectionWeights?.search_score)} / w(lexical)=
                                  {formatScore(selectionWeights?.lexical_score)}
                                </p>
                              </div>
                            ) : null}
                            <a href={ev.source_presigned_url} target="_blank" rel="noreferrer" className="mono">
                              原文を開く
                            </a>
                          </div>
                        );
                      })
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
