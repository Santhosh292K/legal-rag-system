# Nyaya — Frontend

Next.js (App Router, TypeScript, Tailwind v4) chat UI for the Legal RAG
system. Talks to the FastAPI service in [`../backend`](../backend) —
nothing here calls `../rag` directly.

## Setup

```bash
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL, default http://localhost:8000
npm run dev
```

The header's status pill polls `GET /api/health` and reflects the
backend's model-loading state; querying is disabled (with a clear error)
until it reports ready.

## Structure

```
src/
├── app/
│   ├── layout.tsx         Fonts, theme provider, toast host, metadata
│   ├── page.tsx           Chat shell — owns message/case state
│   ├── error.tsx          Route error boundary (Next 16: retry, not reset)
│   ├── not-found.tsx      404 page
│   ├── icon.svg           Branded favicon
│   └── globals.css        Design tokens (color, elevation) + light/dark
├── components/
│   ├── chat/              Message list, answer card, citation chips, IRAC bars
│   ├── cases/             Upload dropzone (case list/creation lives in layout/Sidebar.tsx)
│   ├── history/            HistoryList — past-conversations panel
│   ├── layout/             Header, Sidebar (history + cases), health pill
│   ├── theme/              Dark/light (next-themes) + toast host (sonner)
│   └── ui/                 Badge, Spinner, CopyButton primitives
└── lib/
    ├── api.ts             fetch wrappers for the backend's /api/* routes
    ├── history.ts          Conversation history: localStorage persistence, title/grouping
    ├── types.ts            Types mirroring backend/app/schemas.py
    └── utils.ts
```

## Notes

- Citations are rendered by scanning the answer text for `[SECTION_ID]`
  tokens and matching them against the response's `citations` array
  (`components/chat/AnswerText.tsx`) — chips link to the matching
  citation card further down the same message.
- Answers aren't streamed — the backend call is a single blocking
  request — so the pending state is an indeterminate "thinking" indicator,
  not a real progress bar.
- Case IDs are client-managed: creating one in the sidebar just sets it
  as active locally; it's the backend's `GET /api/cases` (session-scoped,
  in-memory) that's the merge source of truth once a document's been
  uploaded to it.
- Elevation (shadows) is defined once as CSS custom properties in
  `globals.css` (`--shadow-sm`, `--shadow-md`, `--shadow-lg`,
  `--shadow-brand`), referenced with Tailwind's arbitrary-value syntax —
  e.g. `shadow-[var(--shadow-md)]` — rather than Tailwind's own built-in
  `shadow-sm`/`shadow-md` utilities. Keeps dark mode's shadows reading as a
  soft glow instead of an invisible black-on-black smear. Follow that
  pattern for any new elevated surface.
- Toasts (`sonner`, via `components/theme/Toaster.tsx`) are for
  fire-and-forget confirmations (upload succeeded/failed) — chat-level
  errors stay inline in the message thread, not a toast, since they're
  part of the conversation record.
- Conversation history (`lib/history.ts`) is **client-only** — persisted to
  `localStorage` (`nyaya:conversations:v1`), capped at the 50 most
  recent, never sent to the backend. It's per-browser, not per-account:
  clearing site data or switching browsers loses it, and it isn't shared
  across devices. A conversation is only saved once it has a real user
  message (an empty "new chat" is never written), titled from that first
  message's own text (no LLM call). Selecting a history entry restores
  both its messages and whichever case was active at the time.
- All copy shown to end users (empty-state feature list, the in-progress
  "thinking" steps) is deliberately non-technical — this product is used
  by the public and by lawyers, not developers. Never reintroduce
  implementation terms (BM25, IRAC, hybrid/dense retrieval, reranking)
  into user-facing strings; say what it does for them, not how.
