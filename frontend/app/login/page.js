"use client";

import { Suspense, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { setStoredAuth } from "../lib/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const DEFAULT_AUTH_USER = process.env.NEXT_PUBLIC_AUTH_DEV_USER || "admin";
const DEFAULT_AUTH_PASSWORD = process.env.NEXT_PUBLIC_AUTH_DEV_PASSWORD || "admin";

function resolveNextPath(raw) {
  if (!raw || typeof raw !== "string") {
    return "/";
  }
  if (!raw.startsWith("/")) {
    return "/";
  }
  if (raw.startsWith("//")) {
    return "/";
  }
  return raw;
}

function LoginPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const nextPath = useMemo(() => resolveNextPath(searchParams.get("next")), [searchParams]);

  const [username, setUsername] = useState(DEFAULT_AUTH_USER);
  const [password, setPassword] = useState(DEFAULT_AUTH_PASSWORD);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const login = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/auth/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      if (!response.ok) {
        throw new Error(`API ${response.status}: ${await response.text()}`);
      }
      const token = await response.json();
      setStoredAuth(token.access_token || "", token.roles || []);
      router.replace(nextPath);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main>
      <h1>diffUI / Login</h1>
      <p className="lead">Run 詳細の表示には認証が必要です。</p>
      <section className="panel grid">
        <label>
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </label>
        <div className="row">
          <button onClick={login} disabled={busy}>
            ログイン
          </button>
          <button className="secondary" onClick={() => router.replace("/")} disabled={busy}>
            キャンセル
          </button>
        </div>
        <p className="mono">next: {nextPath}</p>
        {error ? <p className="mono" style={{ color: "#b12704" }}>{error}</p> : null}
      </section>
    </main>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<main><p className="mono">loading...</p></main>}>
      <LoginPageContent />
    </Suspense>
  );
}
