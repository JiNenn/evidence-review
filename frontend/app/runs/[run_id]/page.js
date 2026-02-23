"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { clearStoredAuth, getStoredRoles, getStoredToken } from "../../lib/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function fmtDate(value) {
  if (!value) return "-";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString();
}

function fmtDuration(startAt, endAt) {
  if (!startAt) return "-";
  const start = new Date(startAt).getTime();
  const end = endAt ? new Date(endAt).getTime() : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return "-";
  const sec = Math.floor((end - start) / 1000);
  if (sec < 60) return `${sec}s`;
  return `${Math.floor(sec / 60)}m ${sec % 60}s`;
}

function statusLabel(status) {
  const map = {
    pending: "待機中",
    running: "実行中",
    success: "成功",
    success_partial: "一部成功",
    blocked_evidence: "根拠不足",
    failed_system: "システム失敗",
    failed_legacy: "失敗(legacy)"
  };
  return map[status] || status || "-";
}

function statusPillClass(status) {
  if (status === "success" || status === "success_partial") return "status-pill status-pill-ok";
  if (status === "running") return "status-pill status-pill-running";
  if (status === "blocked_evidence" || status === "failed_system" || status === "failed_legacy") {
    return "status-pill status-pill-ng";
  }
  return "status-pill";
}

function preferredTab(status) {
  if (status === "pending") return "inputs";
  if (status === "running") return "pipeline";
  if (status === "success" || status === "success_partial") return "review";
  if (status === "blocked_evidence" || status === "failed_system" || status === "failed_legacy") return "pipeline";
  return "inputs";
}

function shortText(value, limit = 48) {
  const text = String(value || "").trim();
  if (!text) return "";
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}...`;
}

function normTextForMatch(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function rowHasQuery(value, queries) {
  const text = normTextForMatch(value);
  if (!text) return false;
  return (queries || []).some((q) => {
    const needle = normTextForMatch(q);
    return needle.length >= 6 && text.includes(needle);
  });
}

function issueDisplayTitle(title, summary) {
  const rawTitle = String(title || "").trim();
  const rawSummary = String(summary || "").trim();
  const genericTitle = /^(変更検知|追加検知|削除検知)\s*#\d+$/;
  const summaryWithPrefix = /^(変更検知|追加検知|削除検知)\s*:\s*(.+)$/;
  if (genericTitle.test(rawTitle)) {
    const matched = rawSummary.match(summaryWithPrefix);
    if (matched && matched[2]) return matched[2].trim();
  }
  return rawTitle || rawSummary || "論点";
}

function toBulletLines(text, maxItems = 8) {
  const lines = String(text || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) return [];
  return lines.slice(0, maxItems);
}

function toConcreteText(text, maxItems = 3) {
  const lines = toBulletLines(text, maxItems);
  if (lines.length === 0) return "差分テキストなし";
  return lines.join(" / ");
}

function normalizeConcreteExample(text) {
  const raw = String(text || "").trim();
  if (!raw) return "差分テキストなし";
  return raw.replace(/^具体例[:：]\s*/u, "").trim();
}

function toLeftRightGuidance(evidence) {
  const beforeText = evidence?.before_excerpt || "";
  const afterText = evidence?.after_excerpt || "";
  const g = evidence?.writing_guidance && typeof evidence.writing_guidance === "object"
    ? evidence.writing_guidance
    : null;
  const leftPoint = g?.left_point || "書き方のポイント: 前提・対象・制約を省略せず、読み手が現状の意図を誤読しない表現にする。";
  const leftExample = normalizeConcreteExample(g?.left_example || toConcreteText(beforeText));
  const rightPoint = g?.right_point || "書き方のポイント: 結論を先に置き、判断条件と実行アクションを1つの流れで示す。";
  const rightExample = normalizeConcreteExample(g?.right_example || toConcreteText(afterText));
  return [
    {
      id: "left",
      label: "左（現状）",
      point: leftPoint,
      concrete: leftExample
    },
    {
      id: "right",
      label: "右（改善案）",
      point: rightPoint,
      concrete: rightExample
    }
  ];
}

export default function RunConsolePage() {
  const params = useParams();
  const router = useRouter();
  const runId = useMemo(() => String(params?.run_id || ""), [params]);

  const [taskPrompt, setTaskPrompt] = useState("");
  const [runStatus, setRunStatus] = useState("");
  const [issues, setIssues] = useState([]);
  const [issueDetails, setIssueDetails] = useState({});
  const [stages, setStages] = useState([]);
  const [sourceDocs, setSourceDocs] = useState([]);
  const [sourceChunkPreviews, setSourceChunkPreviews] = useState({});

  const [selectedIssueId, setSelectedIssueId] = useState("");
  const [selectedStageKey, setSelectedStageKey] = useState("");
  const [sourceDrawerOpen, setSourceDrawerOpen] = useState(false);
  const [expandedSourceId, setExpandedSourceId] = useState("");
  const [sourceFile, setSourceFile] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [compareData, setCompareData] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareFocus, setCompareFocus] = useState({ issueId: "", beforeQueries: [], afterQueries: [], highlightQueries: [] });

  const [accessToken, setAccessToken] = useState("");
  const [tokenRoles, setTokenRoles] = useState([]);
  const [authHydrated, setAuthHydrated] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [readyStatus, setReadyStatus] = useState({ ok: false, detail: "未取得" });
  const [activeTab, setActiveTab] = useState("inputs");
  const [manualTab, setManualTab] = useState(false);
  const [autoPolling, setAutoPolling] = useState(true);
  const [lastUpdatedAt, setLastUpdatedAt] = useState("");

  const [issueSort, setIssueSort] = useState("severity_desc");
  const [statusFilter, setStatusFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [strengthFilter, setStrengthFilter] = useState("all");

  const toNum = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const redirectToLogin = () => {
    router.replace(`/login?next=${encodeURIComponent(`/runs/${runId}`)}`);
  };

  const openTroubleshoot = () => {
    router.push(`/debug?run_id=${encodeURIComponent(runId)}&return_to=${encodeURIComponent(`/runs/${runId}`)}`);
  };

  const callApi = async (path, options = {}) => {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
    const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
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
      setError("権限不足です。監査権限が必要な操作は `audit/admin` ロールで実行してください。");
      return;
    }
    setError(String(e?.message || e));
  };

  const refreshReady = async () => {
    try {
      const response = await fetch(`${API_BASE}/readyz`);
      if (!response.ok) {
        setReadyStatus({ ok: false, detail: `readyz failed (${response.status})` });
        return;
      }
      setReadyStatus({ ok: true, detail: "ready" });
    } catch (e) {
      setReadyStatus({ ok: false, detail: String(e?.message || e) });
    }
  };

  const refreshAll = async ({ silent = false } = {}) => {
    if (!runId) return;
    if (!silent) {
      setBusy(true);
      setError("");
    }
    try {
      const [run, issueRows, stageRows, sourceRows] = await Promise.all([
        callApi(`/runs/${runId}`),
        callApi(`/runs/${runId}/issues`),
        callApi(`/runs/${runId}/stages`),
        callApi(`/runs/${runId}/sources`)
      ]);
      setTaskPrompt(run.task_prompt || "");
      setRunStatus(run.status || "");
      setIssues(issueRows);
      setStages(stageRows);
      setSourceDocs(sourceRows);
      setLastUpdatedAt(new Date().toISOString());
      await refreshReady();
    } catch (e) {
      handleApiError(e);
    } finally {
      if (!silent) setBusy(false);
    }
  };

  const runInitial = async () => {
    if (!canRun) {
      setError(`実行できません: ${runDisabledReason}`);
      return;
    }
    setBusy(true);
    setError("");
    try {
      await callApi(`/runs/${runId}/pipeline`, { method: "POST", body: JSON.stringify({}) });
      setCompareData(null);
      await refreshAll({ silent: true });
    } catch (e) {
      handleApiError(e);
    } finally {
      setBusy(false);
    }
  };

  const retryFailedStages = async () => {
    setBusy(true);
    setError("");
    try {
      await callApi(`/runs/${runId}/pipeline`, { method: "POST", body: JSON.stringify({}) });
      setCompareData(null);
      await refreshAll({ silent: true });
    } catch (e) {
      handleApiError(e);
    } finally {
      setBusy(false);
    }
  };

  const uploadSource = async () => {
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
        body: JSON.stringify({ filename: sourceFile.name, content_type: contentType })
      });
      const putResp = await fetch(presign.url, {
        method: "PUT",
        headers: { "Content-Type": contentType },
        body: sourceFile
      });
      if (!putResp.ok) {
        throw new Error(`upload failed: ${putResp.status}`);
      }
      await callApi(`/runs/${runId}/sources/ingest`, {
        method: "POST",
        body: JSON.stringify({
          source_doc_id: presign.source_doc_id,
          object_key: presign.object_key,
          title: sourceFile.name,
          content_type: contentType
        })
      });
      setSourceFile(null);
      setFileInputKey((prev) => prev + 1);
      setCompareData(null);
      await refreshAll({ silent: true });
    } catch (e) {
      handleApiError(e);
    } finally {
      setBusy(false);
    }
  };

  const previewChunks = async (sourceDocId) => {
    try {
      const rows = await callApi(`/sources/${sourceDocId}/chunks?limit=20`);
      setSourceChunkPreviews((prev) => ({ ...prev, [sourceDocId]: rows }));
    } catch (e) {
      handleApiError(e);
    }
  };

  const openOriginal = async (sourceDocId) => {
    try {
      const row = await callApi(`/sources/${sourceDocId}/presign-get`);
      window.open(row.url, "_blank", "noopener,noreferrer");
    } catch (e) {
      handleApiError(e);
    }
  };

  const loadCompare = async ({ silent = false } = {}) => {
    if (!runId) return null;
    if (!silent) setCompareLoading(true);
    try {
      const row = await callApi(`/runs/${runId}/compare`);
      setCompareData(row);
      return row;
    } catch (e) {
      if (String(e?.message || "").includes("API 409")) {
        setCompareData(null);
        return null;
      }
      if (!silent) handleApiError(e);
      return null;
    } finally {
      if (!silent) setCompareLoading(false);
    }
  };

  const openIssueDetail = async (issueId) => {
    setSelectedIssueId(issueId);
    if (issueDetails[issueId]) return;
    try {
      const detail = await callApi(`/issues/${issueId}?include_guidance=true`);
      setIssueDetails((prev) => ({ ...prev, [issueId]: detail }));
    } catch (e) {
      handleApiError(e);
    }
  };

  const openIssueCompare = async (issueId) => {
    setSelectedIssueId(issueId);
    try {
      const [detail, context] = await Promise.all([
        issueDetails[issueId] ? Promise.resolve(issueDetails[issueId]) : callApi(`/issues/${issueId}?include_guidance=true`),
        callApi(`/issues/${issueId}/compare-context`)
      ]);
      if (!issueDetails[issueId]) {
        setIssueDetails((prev) => ({ ...prev, [issueId]: detail }));
      }
      setCompareFocus({
        issueId,
        beforeQueries: context.before_queries || [],
        afterQueries: context.after_queries || [],
        highlightQueries: context.highlight_queries || []
      });
      setActiveTab("compare");
      setManualTab(true);
      await loadCompare({ silent: false });
    } catch (e) {
      handleApiError(e);
    }
  };

  const logout = () => {
    clearStoredAuth();
    setAccessToken("");
    setTokenRoles([]);
    router.replace("/");
  };

  useEffect(() => {
    setAccessToken(getStoredToken());
    setTokenRoles(getStoredRoles());
    setAuthHydrated(true);
  }, []);

  useEffect(() => {
    if (!manualTab) setActiveTab(preferredTab(runStatus));
  }, [runStatus, manualTab]);

  useEffect(() => {
    if (!runId || !authHydrated) return;
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, accessToken, authHydrated]);

  useEffect(() => {
    if (!runId || !autoPolling || !authHydrated) return;
    const timer = setInterval(() => refreshAll({ silent: true }), 5000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, autoPolling, accessToken, authHydrated]);

  useEffect(() => {
    if (!runId || !authHydrated || activeTab !== "compare") return;
    if (compareData) return;
    loadCompare({ silent: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, authHydrated, activeTab, accessToken, compareData]);

  const stageRows = useMemo(() => {
    return [...stages].sort((a, b) => new Date(b.started_at || 0).getTime() - new Date(a.started_at || 0).getTime());
  }, [stages]);
  const stageRowKey = (row) => `${row.stage_name}-${row.attempt}-${row.started_at || "na"}`;
  const currentStage = stageRows.find((row) => row.status === "running") || stageRows.find((row) => row.status === "failed") || stageRows[0] || null;
  const selectedStage = selectedStageKey
    ? stageRows.find((row) => stageRowKey(row) === selectedStageKey)
    : currentStage;
  const failedStage = stageRows.find((row) => row.status === "failed") || null;

  const hasIngestedSource = sourceDocs.some((row) => Number(row.chunk_count || 0) > 0);
  const preflight = [
    { key: "source", label: "ソース ingest 済み", ok: hasIngestedSource, detail: hasIngestedSource ? "OK" : "source不足" },
    { key: "ready", label: "DB/Redis/MinIO ready", ok: readyStatus.ok, detail: readyStatus.detail },
    { key: "prompt", label: "task_prompt 非空", ok: Boolean(taskPrompt.trim()), detail: taskPrompt.trim() ? "OK" : "空" },
    { key: "status", label: "Run初回実行可", ok: runStatus === "pending", detail: runStatus || "-" }
  ];
  const blocked = preflight.find((row) => !row.ok);
  const canRun = !blocked;
  const runDisabledReason = blocked
    ? blocked.key === "status"
      ? `初回実行は pending の run のみ実行できます (current=${blocked.detail})`
      : `${blocked.label}: ${blocked.detail}`
    : "";

  const scoreOfIssue = (issue) => {
    const score = Number(issue?.top_evidence_score);
    return Number.isFinite(score) ? score : -1;
  };
  const sevLabel = (sev) => (toNum(sev, 0) >= 3 ? "high" : toNum(sev, 0) >= 2 ? "medium" : "low");
  const selectedIssue = selectedIssueId ? issues.find((row) => row.id === selectedIssueId) : null;
  const selectedIssueDetail = selectedIssue ? issueDetails[selectedIssue.id] : null;
  const evidenceStrength = (evidence) => {
    const selection =
      evidence?.selection && typeof evidence.selection === "object"
        ? evidence.selection
        : evidence?.citation_span && typeof evidence.citation_span === "object"
          ? evidence.citation_span.selection
          : null;
    if (!selection || typeof selection !== "object") {
      return { combined: -1, rank: Number.MAX_SAFE_INTEGER };
    }
    return {
      combined: toNum(selection.combined_score, -1),
      rank: toNum(selection.search_rank, Number.MAX_SAFE_INTEGER)
    };
  };
  const sortedEvidences = selectedIssueDetail
    ? [...(selectedIssueDetail.evidences || [])].sort((a, b) => {
        const ax = evidenceStrength(a);
        const bx = evidenceStrength(b);
        if (bx.combined !== ax.combined) return bx.combined - ax.combined;
        if (ax.rank !== bx.rank) return ax.rank - bx.rank;
        return String(a.id).localeCompare(String(b.id));
      })
    : [];
  const visibleIssues = [...issues]
    .filter((row) => statusFilter === "all" || String(row.status) === statusFilter)
    .filter((row) => severityFilter === "all" || sevLabel(row.severity) === severityFilter)
    .filter((row) => {
      if (strengthFilter === "all") return true;
      const score = scoreOfIssue(row);
      if (score < 0) return strengthFilter === "unknown";
      if (strengthFilter === "strong") return score >= 0.7;
      if (strengthFilter === "medium") return score >= 0.4 && score < 0.7;
      if (strengthFilter === "weak") return score < 0.4;
      return true;
    })
    .sort((a, b) => {
      if (issueSort === "evidence_desc") return scoreOfIssue(b) - scoreOfIssue(a);
      if (issueSort === "confidence_desc") return toNum(b.confidence, 0) - toNum(a.confidence, 0);
      if (issueSort === "updated_desc") return new Date(b.updated_at || 0).getTime() - new Date(a.updated_at || 0).getTime();
      if (issueSort === "status_open_first") {
        const order = { open: 0, resolved: 1, hidden: 2 };
        return (order[String(a.status)] ?? 9) - (order[String(b.status)] ?? 9);
      }
      return toNum(b.severity, 0) - toNum(a.severity, 0);
    });

  const canRetryFailedStages = Boolean(failedStage) && runStatus === "failed_system";
  const canTroubleshoot = runStatus === "blocked_evidence" || runStatus === "failed_system" || Boolean(failedStage);
  const followStatus = !manualTab;
  const shortRunId = runId ? `${runId.slice(0, 8)}...${runId.slice(-4)}` : "-";
  const compactStage = shortText(currentStage ? currentStage.stage_name : "-", 32);
  const canStartInitialRun = runStatus === "pending";
  const showCancelAction = runStatus === "running";
  const showRunDisabledReason = canStartInitialRun && !canRun;
  const statusFollowHelp = "Run状態に応じて Inputs/Pipeline/Review を自動切替（Compareは手動）";

  const issuesEmptyMessage = () => {
    if (runStatus === "success") return "内容がほとんど一致しています。";
    if (runStatus === "blocked_evidence") return "根拠不足のため論点を確定できませんでした。Debugで復旧してください。";
    if (runStatus === "running" || runStatus === "pending") return "Review は生成中です。";
    return "表示可能な論点はまだありません。";
  };

  const compareRows = compareData?.rows || [];
  const compareFocusTerms = [
    ...(compareFocus.beforeQueries || []),
    ...(compareFocus.afterQueries || []),
    ...(compareFocus.highlightQueries || [])
  ];

  return (
    <main className="run-console">
      <section className="panel run-console-header">
        <div className="run-header-top">
          <div className="run-title-row">
            <h1 className="run-title">Run {shortRunId}</h1>
            <span className={statusPillClass(runStatus)}>{statusLabel(runStatus)}</span>
          </div>
          <div className="row run-header-links">
            <a className="mono" href="/">Home</a>
          </div>
        </div>

        <div className="row run-meta-row">
          <span className="chip" title={runId}>id: {shortRunId}</span>
          <span className="chip">roles: {tokenRoles.join(",") || "-"}</span>
        </div>

        <div className="mono run-meta-line">
          stage={compactStage} | elapsed=
          {currentStage ? fmtDuration(currentStage.started_at, currentStage.finished_at) : "-"} | retry=
          {currentStage ? Math.max(0, (currentStage.attempt || 1) - 1) : 0} | updated={fmtDate(lastUpdatedAt)}
        </div>

        <div className="run-actions-row">
          <div className="row">
            {canStartInitialRun ? (
              <button className="compact-btn" onClick={runInitial} disabled={busy || !canRun}>Run（初回実行）</button>
            ) : null}
            {showCancelAction ? <button className="secondary compact-btn" disabled title="Cancel API は未実装">Cancel</button> : null}
            {canRetryFailedStages ? <button className="secondary compact-btn" onClick={retryFailedStages} disabled={busy}>Retry failed stages</button> : null}
            <button className="secondary compact-btn" onClick={openTroubleshoot}>{runStatus === "blocked_evidence" ? "Debugで復旧" : "Open Debug"}</button>
            <button className="secondary compact-btn" onClick={logout}>ログアウト</button>
          </div>
          <div className="row run-settings-row">
            <label className="toggle-inline mono" title={statusFollowHelp}>
              <input
                type="checkbox"
                checked={followStatus}
                onChange={(e) => {
                  const enabled = e.target.checked;
                  setManualTab(!enabled);
                  if (enabled) setActiveTab(preferredTab(runStatus));
                }}
              />
              状態に追従
            </label>
            <label className="toggle-inline mono">
              <input type="checkbox" checked={autoPolling} onChange={(e) => setAutoPolling(e.target.checked)} />
              自動更新
            </label>
          </div>
        </div>

        {showRunDisabledReason ? <p className="mono status-note">Run disabled: {runDisabledReason}</p> : null}
        {error ? <p className="mono status-ng">{error}</p> : null}
      </section>

      <section className="panel console-tabs" style={{ marginTop: 14 }}>
        <nav className="tabs" role="tablist" aria-label="Run Console Tabs">
          <button
            type="button"
            id="tab-inputs"
            role="tab"
            aria-controls="panel-inputs"
            aria-selected={activeTab === "inputs"}
            className={`tab ${activeTab === "inputs" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("inputs");
              setManualTab(true);
            }}
          >
            Inputs
          </button>
          <button
            type="button"
            id="tab-pipeline"
            role="tab"
            aria-controls="panel-pipeline"
            aria-selected={activeTab === "pipeline"}
            className={`tab ${activeTab === "pipeline" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("pipeline");
              setManualTab(true);
            }}
          >
            Pipeline
          </button>
          <button
            type="button"
            id="tab-compare"
            role="tab"
            aria-controls="panel-compare"
            aria-selected={activeTab === "compare"}
            className={`tab ${activeTab === "compare" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("compare");
              setManualTab(true);
            }}
          >
            Compare
          </button>
          <button
            type="button"
            id="tab-review"
            role="tab"
            aria-controls="panel-review"
            aria-selected={activeTab === "review"}
            className={`tab ${activeTab === "review" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("review");
              setManualTab(true);
            }}
          >
            Review
          </button>
        </nav>
      </section>

      {activeTab === "inputs" ? (
        <section role="tabpanel" id="panel-inputs" aria-labelledby="tab-inputs" className="grid" style={{ marginTop: 14 }}>
          <article className="panel">
            <h2>Preflight Checklist</h2>
            {preflight.map((row) => <p key={row.key} className="mono">{row.ok ? "OK" : "NG"} {row.label} / {row.detail}</p>)}
            <p className="mono">予想実行時間: {Math.max(15, sourceDocs.length * 20)}s 目安</p>
          </article>
          <article className="panel">
            <div className="row">
              <h2 style={{ margin: 0 }}>Sources</h2>
              <button className="secondary" onClick={() => setSourceDrawerOpen(true)}>Add sources</button>
            </div>
            {sourceDocs.length === 0 ? <p className="mono">ソースはまだありません。</p> : (
              <div style={{ overflowX: "auto" }}>
                <table className="stage-table sources-table">
                  <thead>
                    <tr>
                      <th>title</th>
                      <th>status</th>
                      <th>chunks</th>
                      <th>created</th>
                      <th>actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sourceDocs.map((doc) => (
                      <Fragment key={doc.id}>
                        <tr>
                          <td>
                            <div>{doc.title}</div>
                            <div className="mono subtle">{doc.content_type}</div>
                          </td>
                          <td>{doc.ingest_status}</td>
                          <td>{doc.chunk_count}</td>
                          <td>{fmtDate(doc.created_at)}</td>
                          <td>
                            <div className="row">
                              <button
                                type="button"
                                className="secondary"
                                onClick={() => setExpandedSourceId((prev) => (prev === doc.id ? "" : doc.id))}
                              >
                                {expandedSourceId === doc.id ? "閉じる" : "詳細"}
                              </button>
                              <button type="button" className="secondary" onClick={() => previewChunks(doc.id)}>
                                Preview
                              </button>
                              <button type="button" className="secondary" onClick={() => openOriginal(doc.id)}>
                                Open
                              </button>
                            </div>
                          </td>
                        </tr>
                        {expandedSourceId === doc.id ? (
                          <tr className="sources-detail-row">
                            <td colSpan={5}>
                              <p className="mono">object_key={doc.object_key}</p>
                              {(sourceChunkPreviews[doc.id] || []).length === 0 ? (
                                <p className="mono">chunk preview 未取得</p>
                              ) : (
                                (sourceChunkPreviews[doc.id] || []).map((chunk) => (
                                  <div key={chunk.id} className="panel" style={{ marginTop: 6 }}>
                                    <p className="mono">chunk_index={chunk.chunk_index}</p>
                                    <pre>{chunk.text_excerpt}</pre>
                                  </div>
                                ))
                              )}
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        </section>
      ) : null}

      {activeTab === "pipeline" ? (
        <section role="tabpanel" id="panel-pipeline" aria-labelledby="tab-pipeline" className="grid" style={{ marginTop: 14 }}>
          <article className="panel">
            <h2>Stages</h2>
            {stageRows.length === 0 ? <p className="mono">stage情報はまだありません。</p> : (
              <div style={{ overflowX: "auto" }}>
                <table className="stage-table">
                  <thead><tr><th>status</th><th>stage</th><th>elapsed</th><th>attempt</th><th>output</th></tr></thead>
                  <tbody>
                    {stageRows.map((stage) => {
                      const key = stageRowKey(stage);
                      const active = selectedStage && stageRowKey(selectedStage) === key;
                      return (
                        <tr key={key} className={active ? "active" : ""} onClick={() => setSelectedStageKey(key)}>
                          <td>{stage.status}</td>
                          <td>{stage.stage_name}</td>
                          <td>{fmtDuration(stage.started_at, stage.finished_at)}</td>
                          <td>{stage.attempt}</td>
                          <td>{stage.output_ref && Object.keys(stage.output_ref).length > 0 ? "available" : "-"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {selectedStage ? (
              <div className="panel" style={{ marginTop: 10 }}>
                <h3 style={{ marginTop: 0 }}>Stage Detail</h3>
                <p className="mono">{selectedStage.stage_name} / status={selectedStage.status} / attempt={selectedStage.attempt}</p>
                {selectedStage.failure_type ? <p className="mono">failure_type={selectedStage.failure_type}</p> : null}
                {selectedStage.failure_detail ? <pre>{JSON.stringify(selectedStage.failure_detail, null, 2)}</pre> : null}
                {selectedStage.output_ref ? <pre>{JSON.stringify(selectedStage.output_ref, null, 2)}</pre> : null}
              </div>
            ) : null}
          </article>

          {canTroubleshoot ? (
            <article className="panel">
              <h2>Troubleshoot</h2>
              <p className="mono">failure_type={failedStage?.failure_type || "-"} / summary={failedStage?.failure_detail?.summary || failedStage?.output_ref?.error || "-"}</p>
              <div className="row">
                <button className="secondary" onClick={() => { setActiveTab("inputs"); setSourceDrawerOpen(true); }}>ソース追加</button>
                {canRetryFailedStages ? (
                  <button className="secondary" onClick={retryFailedStages} disabled={busy}>
                    Retry failed stages
                  </button>
                ) : null}
              </div>
              <p className="mono subtle">詳細調査はヘッダーの Open Debug から実行できます。</p>
            </article>
          ) : null}
        </section>
      ) : null}

      {activeTab === "compare" ? (
        <section role="tabpanel" id="panel-compare" aria-labelledby="tab-compare" className="grid" style={{ marginTop: 14 }}>
          <article className="panel">
            <div className="row compare-toolbar">
              <h2 style={{ margin: 0 }}>Compare (左: High one-shot / 右: Low+FB)</h2>
              <button className="secondary" onClick={() => loadCompare({ silent: false })} disabled={busy || compareLoading}>
                再読込
              </button>
            </div>
            {compareFocus.issueId ? (
              <p className="mono">
                issue focus: {shortText(compareFocus.issueId, 24)} / terms={compareFocusTerms.length}
              </p>
            ) : (
              <p className="mono subtle">Review の Issue Detail から「比較で確認」を押すと関連箇所をハイライトします。</p>
            )}
            {compareLoading ? (
              <p className="mono">比較データを読み込み中...</p>
            ) : !compareData ? (
              <p className="mono">比較データはまだありません。Pipeline 完了後に再読込してください。</p>
            ) : (
              <div className="compare-table-wrap">
                <table className="compare-table">
                  <thead>
                    <tr>
                      <th className="ln-col">L#</th>
                      <th>{compareData.left_label || "High model one-shot"}</th>
                      <th className="ln-col">R#</th>
                      <th>{compareData.right_label || "Low draft + high feedback"}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compareRows.length === 0 ? (
                      <tr>
                        <td colSpan={4}>
                          <p className="mono">比較対象の本文が空です。</p>
                        </td>
                      </tr>
                    ) : compareRows.map((row) => {
                      const leftHit = rowHasQuery(row.left_text, compareFocusTerms);
                      const rightHit = rowHasQuery(row.right_text, compareFocusTerms);
                      const focusClass = leftHit || rightHit ? "is-focus" : "";
                      return (
                        <tr key={row.row_no} className={`cmp-row cmp-${row.kind} ${focusClass}`}>
                          <td className="ln-col mono">{row.left_line_no || ""}</td>
                          <td className={leftHit ? "cmp-hit-side" : ""}>{row.left_text || "\u00A0"}</td>
                          <td className="ln-col mono">{row.right_line_no || ""}</td>
                          <td className={rightHit ? "cmp-hit-side" : ""}>{row.right_text || "\u00A0"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        </section>
      ) : null}

      {activeTab === "review" ? (
        <section role="tabpanel" id="panel-review" aria-labelledby="tab-review" className="grid" style={{ marginTop: 14 }}>
          <article className="panel review-layout">
            <div>
              <h2>Issues</h2>
              <div className="row" style={{ marginBottom: 10 }}>
                <select value={issueSort} onChange={(e) => setIssueSort(e.target.value)}>
                  <option value="severity_desc">並び順: 影響度</option>
                  <option value="evidence_desc">並び順: 根拠強度</option>
                  <option value="confidence_desc">並び順: confidence</option>
                  <option value="updated_desc">並び順: 更新時刻</option>
                  <option value="status_open_first">並び順: 未解決優先</option>
                </select>
                <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                  <option value="all">状態: すべて</option>
                  <option value="open">状態: open</option>
                  <option value="resolved">状態: resolved</option>
                </select>
                <select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}>
                  <option value="all">影響度: すべて</option>
                  <option value="high">影響度: high</option>
                  <option value="medium">影響度: medium</option>
                  <option value="low">影響度: low</option>
                </select>
                <select value={strengthFilter} onChange={(e) => setStrengthFilter(e.target.value)}>
                  <option value="all">根拠強度: すべて</option>
                  <option value="strong">根拠強度: strong (&gt;=0.7)</option>
                  <option value="medium">根拠強度: medium (0.4-0.7)</option>
                  <option value="weak">根拠強度: weak (&lt;0.4)</option>
                  <option value="unknown">根拠強度: unknown</option>
                </select>
              </div>

              {issues.length === 0 ? <p className="mono">{issuesEmptyMessage()}</p> : visibleIssues.length === 0 ? <p className="mono">現在のフィルタ条件に一致する論点はありません。</p> : visibleIssues.map((issue) => (
                <div key={issue.id} className="panel" style={{ marginTop: 8 }}>
                  <p><strong>{issueDisplayTitle(issue.title, issue.summary)}</strong></p>
                  <p className="mono subtle">{issue.summary}</p>
                  <p className="mono">severity={sevLabel(issue.severity)}({issue.severity}) / confidence={toNum(issue.confidence, 0).toFixed(2)} / status={issue.status}</p>
                  <p className="mono">evidences={issue.evidence_count} / top_evidence_score={toNum(issue.top_evidence_score, -1).toFixed(3)}</p>
                  <div className="row">
                    <button className="secondary" onClick={() => openIssueDetail(issue.id)}>右ペインで表示</button>
                    <button className="secondary" onClick={() => openIssueCompare(issue.id)}>比較で確認</button>
                  </div>
                </div>
              ))}
            </div>

            <div className="panel">
              <h3 style={{ marginTop: 0 }}>Issue Detail</h3>
              {!selectedIssue ? <p className="mono">Issue を選択してください。</p> : !selectedIssueDetail ? <p className="mono">読み込み中...</p> : (
                <>
                  <p><strong>{issueDisplayTitle(selectedIssueDetail.title, selectedIssueDetail.summary)}</strong></p>
                  <p className="mono subtle">{selectedIssueDetail.summary}</p>
                  <p className="mono">severity={sevLabel(selectedIssueDetail.severity)}({selectedIssueDetail.severity}) / confidence={toNum(selectedIssueDetail.confidence, 0).toFixed(2)}</p>
                  <button className="secondary" onClick={() => openIssueCompare(selectedIssueDetail.id)}>比較ビューへ移動</button>
                  {sortedEvidences.length === 0 ? <p className="mono">この論点には表示可能な根拠がありません。</p> : sortedEvidences.map((ev) => (
                    <div key={ev.id} className="panel" style={{ marginTop: 8 }}>
                      <p className="mono">source: {ev.source_title}</p>
                      <ol className="issue-numbered-list">
                        {toLeftRightGuidance(ev).map((item) => (
                          <li key={`${ev.id}-${item.id}`}>
                            <p className="issue-guidance-label">{item.label}</p>
                            <p>{item.point}</p>
                            <p className="mono subtle"># {item.concrete}</p>
                          </li>
                        ))}
                      </ol>
                      <a href={ev.source_presigned_url} target="_blank" rel="noreferrer" className="mono">原文を開く</a>
                    </div>
                  ))}
                </>
              )}
            </div>
          </article>
        </section>
      ) : null}

      {sourceDrawerOpen ? (
        <div className="drawer-backdrop" onClick={() => setSourceDrawerOpen(false)}>
          <div className="drawer" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>Add Sources</h3>
            <p className="mono">複数ファイル対応は次段で実装。現在は1件ずつ取り込みます。</p>
            <input key={fileInputKey} type="file" onChange={(e) => setSourceFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)} />
            <div className="row" style={{ marginTop: 8 }}>
              <button onClick={uploadSource} disabled={busy || !sourceFile}>アップロードして取り込み</button>
              <button className="secondary" onClick={() => setSourceDrawerOpen(false)}>閉じる</button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
