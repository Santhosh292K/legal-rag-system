"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef } from "react";
import type { ChatMessage } from "@/lib/types";
import { Composer } from "./Composer";
import { EmptyState } from "./EmptyState";
import { MessageRow } from "./MessageRow";

export function ChatPanel({
  messages,
  onSend,
  sending,
  activeCaseId,
}: {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  sending: boolean;
  activeCaseId: string | null;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <AnimatePresence mode="wait" initial={false}>
        {messages.length === 0 ? (
          <motion.div
            key="empty"
            className="flex flex-1 flex-col"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <EmptyState onPick={onSend} />
          </motion.div>
        ) : (
          <motion.div
            key="thread"
            className="flex-1 overflow-y-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <div className="mx-auto flex max-w-3xl flex-col gap-5 px-4 py-6 sm:px-6">
              {messages.map((m) => (
                <MessageRow key={m.id} message={m} />
              ))}
              <div ref={bottomRef} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <Composer onSend={onSend} disabled={sending} activeCaseId={activeCaseId} />
    </div>
  );
}
