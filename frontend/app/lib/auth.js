"use client";

const TOKEN_KEY = "diffui_access_token";
const ROLES_KEY = "diffui_access_roles";

export function getStoredToken() {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(TOKEN_KEY) || "";
}

export function getStoredRoles() {
  if (typeof window === "undefined") {
    return [];
  }
  const raw = window.localStorage.getItem(ROLES_KEY) || "[]";
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function setStoredAuth(token, roles = []) {
  if (typeof window === "undefined") {
    return;
  }
  if (!token) {
    window.localStorage.removeItem(TOKEN_KEY);
  } else {
    window.localStorage.setItem(TOKEN_KEY, token);
  }
  window.localStorage.setItem(ROLES_KEY, JSON.stringify(Array.isArray(roles) ? roles : []));
}

export function clearStoredAuth() {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(ROLES_KEY);
}
