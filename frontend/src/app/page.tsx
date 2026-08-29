"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, listCases, postQuery } from "@/lib/api";
import { deriveTitle, loadConversations, saveConversations } from "@/lib/history";
import type { Conversation } from "@/lib/history";
import type { CaseInfo, ChatMessage } from "@/lib/types";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
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

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [historyLoaded, setHistoryLoaded] = useState(false);

  // localStorage is only available client-side, so this has to be an
  // effect rather than useState's initializer (which would run during SSR
  // and desync from what the client then loads on hydration).
  useEffect(() => {
    // Reading localStorage is inherently a client-only side effect; this
    // is the one-time hydration read, not a reactive response to a
    // dependency.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setConversations(loadConversations());
    setHistoryLoaded(true);
  }, []);

  // Persist on every change, once the initial load has happened (guards
  // against briefly overwriting localStorage with [] before the load
  // effect above has run).
  useEffect(() => {
    if (!historyLoaded) return;
    saveConversations(conversations);
  }, [conversations, historyLoaded]);

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

  function rememberCase(id: string) {
    setLocalCaseIds((prev) => (prev.includes(id) ? prev : [...prev, id]));
    setCases((prev) =>
      prev.some((c) => c.case_id === id) ? prev : [...prev, { case_id: id, documents: [] }],
    );
  }

  function handleCreateCase(id: string) {
    rememberCase(id);
    setActiveCaseId(id);
    setSidebarOpen(false);
  }

  function handleSelectCase(id: string | null) {
    setActiveCaseId(id);
    setSidebarOpen(false);
  }

  function handleNewChat() {
    setMessages([]);
    setActiveConversationId(null);
    setSidebarOpen(false);
  }

  function handleSelectConversation(id: string) {
    const conv = conversations.find((c) => c.id === id);
    if (!conv) return;
    setMessages(conv.messages);
    setActiveConversationId(id);
    setActiveCaseId(conv.caseId);
    if (conv.caseId) rememberCase(conv.caseId);
    setSidebarOpen(false);
  }

  function handleDeleteConversation(id: string) {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (id === activeConversationId) {
      setMessages([]);
      setActiveConversationId(null);
    }
  }

  /** Saves the current thread into `conversations` under `convId` —
   * called synchronously at each point handleSend's own `messages` value
   * changes, rather than reactively off a `messages` effect, so it can't
   * race with — or double-fire alongside — the setMessages calls themselves. */
  function upsertConversation(msgs: ChatMessage[], convId: string, caseId: string | null) {
    const firstUserMsg = msgs.find((m) => m.role === "user");
    if (!firstUserMsg) return;
    setConversations((prev) => {
      const existing = prev.find((c) => c.id === convId);
      const title = existing?.title ?? deriveTitle(firstUserMsg.text);
      const updated: Conversation = {
        id: convId,
        title,
        messages: msgs,
        caseId,
        updatedAt: Date.now(),
      };
      return [updated, ...prev.filter((c) => c.id !== convId)];
    });
  }

  async function handleSend(text: string) {
    const convId = activeConversationId ?? uid();
    if (!activeConversationId) setActiveConversationId(convId);

    const userMsg: ChatMessage = { id: uid(), role: "user", text, caseId: activeCaseId };
    const pendingId = uid();
    const withPending: ChatMessage[] = [
      ...messages,
      userMsg,
      { id: pendingId, role: "pending", text },
    ];
    setMessages(withPending);
    upsertConversation(withPending, convId, activeCaseId);
    setSending(true);

    try {
      const answer = await postQuery(text, activeCaseId);
      const resolved: ChatMessage[] = withPending.map((m) =>
        m.id === pendingId ? { id: pendingId, role: "assistant", answer } : m,
      );
      setMessages(resolved);
      upsertConversation(resolved, convId, activeCaseId);
    } catch (e) {
      const detail = e instanceof ApiError ? e.message : "Something went wrong.";
      const resolved: ChatMessage[] = withPending.map((m) =>
        m.id === pendingId ? { id: pendingId, role: "error", text: detail } : m,
      );
      setMessages(resolved);
      upsertConversation(resolved, convId, activeCaseId);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Header
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        onNewChat={handleNewChat}
        hasMessages={messages.length > 0}
      />
      <div className="relative flex min-h-0 flex-1">
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-30 bg-black/30 backdrop-blur-[1px] lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
        <Sidebar
          cases={cases}
          activeCaseId={activeCaseId}
          onSelectCase={handleSelectCase}
          onCreateCase={handleCreateCase}
          onUploaded={refreshCases}
          open={sidebarOpen}
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={handleSelectConversation}
          onDeleteConversation={handleDeleteConversation}
          onNewChat={handleNewChat}
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
