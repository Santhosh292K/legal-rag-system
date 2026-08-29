"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    // Standard next-themes hydration guard: the server can't know the
    // resolved theme, so render a neutral placeholder until mounted.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  const isDark = mounted && resolvedTheme === "dark";

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label="Toggle color theme"
      className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-border text-text-muted transition-colors hover:border-border-strong hover:text-text hover:bg-bg-subtle"
    >
      {mounted ? (
        isDark ? <Sun size={15} /> : <Moon size={15} />
      ) : (
        <span className="block h-[15px] w-[15px]" />
      )}
    </button>
  );
}
