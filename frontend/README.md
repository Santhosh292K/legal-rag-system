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
├── app/                  Route: layout.tsx (fonts, theme), page.tsx (chat shell)
├── components/
│   ├── chat/             Message list, answer card, citation chips, IRAC bars
│   ├── cases/             Case sidebar, upload dropzone
│   ├── layout/             Header, health pill
│   ├── theme/              Dark/light (next-themes)
│   └── ui/                 Badge, Spinner primitives
└── lib/
    ├── api.ts             fetch wrappers for the backend's /api/* routes
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
