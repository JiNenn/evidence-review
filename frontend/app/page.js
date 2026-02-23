"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function formatDate(value) {
  if (!value) {
    return "-";
  }
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    return value;
  }
  return d.toLocaleString();
}

function statusChipClass(status) {
  if (status === "success" || status === "success_partial") {
    return "chip status-ok";
  }
  if (status === "blocked_evidence" || status === "failed_system" || status === "failed_legacy") {
    return "chip status-ng";
  }
  return "chip";
}

export default function HomePage() {
  const router = useRouter();
  const [runs, setRuns] = useState([]);
  const [taskPrompt, setTaskPrompt] = useState("根拠付きでレビュー論点を抽出してください。");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadRuns = async () => {
    const response = await fetch(`${API_BASE}/runs`);
    if (!response.ok) {
      throw new Error(`API ${response.status}: ${await response.text()}`);
    }
    const rows = await response.json();
    setRuns(Array.isArray(rows) ? rows : []);
  };

  const createRun = async () => {
    if (!taskPrompt.trim()) {
      setError("task prompt を入力してください。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_prompt: taskPrompt.trim(),
          metadata: { created_from: "home" }
        })
      });
      if (!response.ok) {
        throw new Error(`API ${response.status}: ${await response.text()}`);
      }
      const run = await response.json();
      router.push(`/runs/${run.id}`);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const refreshRuns = async () => {
    setBusy(true);
    setError("");
    try {
      await loadRuns();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    let active = true;
    const bootstrap = async () => {
      setBusy(true);
      setError("");
      try {
        await loadRuns();
      } catch (e) {
        if (active) {
          setError(String(e.message || e));
        }
      } finally {
        if (active) {
          setBusy(false);
        }
      }
    };
    bootstrap();
    return () => {
      active = false;
    };
  }, []);

  return (
    <main>
      <h1>diffUI / Home</h1>
      <p className="lead">Home と Run 一覧を統合した画面。主導線は Run 詳細です。</p>

      <section className="panel grid">
        <h2>New Run</h2>
        <label>
          Task Prompt
          <textarea rows={4} value={taskPrompt} onChange={(e) => setTaskPrompt(e.target.value)} />
        </label>
        <div className="row">
          <button onClick={createRun} disabled={busy}>
            Run 作成
          </button>
          <button className="secondary" onClick={refreshRuns} disabled={busy}>
            一覧更新
          </button>
          <button className="secondary" onClick={() => router.push("/debug")} disabled={busy}>
            Debug へ
          </button>
        </div>
        {error ? <p className="mono" style={{ color: "#b12704" }}>{error}</p> : null}
      </section>

      <section className="panel grid" style={{ marginTop: 16 }}>
        <h2>Run List</h2>
        {runs.length === 0 ? (
          <p className="mono">Run はまだありません。</p>
        ) : (
          runs.map((run) => (
            <div key={run.id} className="panel" style={{ marginTop: 8 }}>
              <p>
                <strong>{run.title}</strong>
              </p>
              <p className="mono">
                <span className={statusChipClass(run.status)}>status: {run.status}</span>
                <span className="chip">created: {formatDate(run.created_at)}</span>
                <span className="chip">sources: {run.source_count}</span>
                <span className="chip">issues: {run.visible_issue_count}/{run.issue_count}</span>
              </p>
              <div className="row">
                <button className="secondary" onClick={() => router.push(`/runs/${run.id}`)}>
                  詳細
                </button>
              </div>
            </div>
          ))
        )}
      </section>
    </main>
  );
}
