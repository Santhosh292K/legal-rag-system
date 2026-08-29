"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, listCases, postQuery } from "@/lib/api";
import type { CaseInfo, ChatMessage } from "@/lib/types";
import { Header } from "@/components/layout/Header";
import { CaseSidebar } from "@/components/cases/CaseSidebar";
import { ChatPanel } from "@/components/chat/ChatPanel";

function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);
  const [cases, setCases] = useState<CaseInfo[]>([]);
  const [localCaseIds, setLocalCaseIds] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refreshCases = useCallback(async () => {
    try {
      const fetched = await listCases();
      const byId = new Map(fetched.map((c) => [c.case_id, c]));
      for (const id of localCaseIds) {
        if (!byId.has(id)) byId.set(id, { case_id: id, documents: [] });
      }
      setCases(Array.from(byId.values()));
    } catch {
      // Backend not reachable yet — the header's health pill already
      // surfaces this; the case list just stays as-is until it recovers.
    }
  }, [localCaseIds]);

  useEffect(() => {
    // refreshCases is async — its setState call happens after the
    // `await`, not synchronously within this effect body.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshCases();
  }, [refreshCases]);

  function handleCreateCase(id: string) {
    setLocalCaseIds((prev) => (prev.includes(id) ? prev : [...prev, id]));
    setCases((prev) =>
      prev.some((c) => c.case_id === id) ? prev : [...prev, { case_id: id, documents: [] }],
    );
    setActiveCaseId(id);
    setSidebarOpen(false);
  }

  function handleSelectCase(id: string | null) {
    setActiveCaseId(id);
    setSidebarOpen(false);
  }

  async function handleSend(text: string) {
    const userMsg: ChatMessage = { id: uid(), role: "user", text, caseId: activeCaseId };
    const pendingId = uid();
    setMessages((prev) => [...prev, userMsg, { id: pendingId, role: "pending", text }]);
    setSending(true);

    try {
      const answer = await postQuery(text, activeCaseId);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId ? { id: pendingId, role: "assistant", answer } : m,
        ),
      );
    } catch (e) {
      const detail = e instanceof ApiError ? e.message : "Something went wrong.";
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { id: pendingId, role: "error", text: detail } : m)),
      );
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Header onToggleSidebar={() => setSidebarOpen((v) => !v)} />
      <div className="relative flex min-h-0 flex-1">
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-30 bg-black/30 backdrop-blur-[1px] lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
        <CaseSidebar
          cases={cases}
          activeCaseId={activeCaseId}
          onSelectCase={handleSelectCase}
          onCreateCase={handleCreateCase}
          onUploaded={refreshCases}
          open={sidebarOpen}
        />
        <ChatPanel
          messages={messages}
          onSend={handleSend}
          sending={sending}
          activeCaseId={activeCaseId}
        />
      </div>
    </div>
  );
}
