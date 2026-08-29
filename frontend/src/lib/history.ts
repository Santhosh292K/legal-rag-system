import type { ChatMessage } from "./types";

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  caseId: string | null;
  updatedAt: number;
}

const STORAGE_KEY = "nyaya:conversations:v1";
const MAX_CONVERSATIONS = 50;
const TITLE_MAX_LEN = 56;

/** Derives a conversation title from its first user message — no LLM
 * call, just a truncation. Matches the question a user actually asked,
 * which is what they'll recognize the conversation by later. */
export function deriveTitle(text: string): string {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > TITLE_MAX_LEN ? `${clean.slice(0, TITLE_MAX_LEN).trimEnd()}…` : clean;
}

export function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    // Corrupted/foreign data in this key, or storage disabled — start
    // fresh rather than throwing and breaking the whole app on load.
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  if (typeof window === "undefined") return;
  try {
    // Keep only the most recent MAX_CONVERSATIONS — this is a convenience
    // history, not a permanent record; unbounded growth in localStorage
    // (a few MB quota, shared with everything else on the origin) isn't
    // worth it for chats nobody's opened in months.
    const trimmed = [...conversations]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_CONVERSATIONS);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch {
    // Quota exceeded or storage disabled (private browsing, etc.) —
    // history just won't persist this session; not worth surfacing an
    // error for a convenience feature.
  }
}

type DateBucket = "Today" | "Yesterday" | "Previous 7 days" | "Older";

export function bucketOf(updatedAt: number, now: number = Date.now()): DateBucket {
  const day = 24 * 60 * 60 * 1000;
  const startOfToday = new Date(now).setHours(0, 0, 0, 0);
  const startOfYesterday = startOfToday - day;
  const startOfWeek = startOfToday - 7 * day;

  if (updatedAt >= startOfToday) return "Today";
  if (updatedAt >= startOfYesterday) return "Yesterday";
  if (updatedAt >= startOfWeek) return "Previous 7 days";
  return "Older";
}

export function groupByRecency(conversations: Conversation[]): [DateBucket, Conversation[]][] {
  const order: DateBucket[] = ["Today", "Yesterday", "Previous 7 days", "Older"];
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const groups = new Map<DateBucket, Conversation[]>();
  for (const c of sorted) {
    const bucket = bucketOf(c.updatedAt);
    const list = groups.get(bucket) ?? [];
    list.push(c);
    groups.set(bucket, list);
  }
  return order.filter((b) => groups.has(b)).map((b) => [b, groups.get(b)!]);
}
