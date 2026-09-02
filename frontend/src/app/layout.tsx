import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Source_Serif_4 } from "next/font/google";
import { ThemeProvider } from "@/components/theme/ThemeProvider";
import { Toaster } from "@/components/theme/Toaster";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Editorial serif for the wordmark, headings, and citation act names — see
// globals.css's --font-serif token for why (ink & brass identity, not the
// default sans-only "AI chat product" look). Weights kept to what's
// actually used (medium/semibold for headings, regular italic for the
// act-name byline on each citation card) rather than pulling the whole family.
const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  style: ["normal", "italic"],
});

const title = "Nyaya — Legal Research Assistant";
const description =
  "Citation-grounded Q&A over Indian statute law — hybrid retrieval, IRAC reranking, and temporal validity filtering, with optional per-case document fusion.";

export const metadata: Metadata = {
  title: {
    default: title,
    template: "%s · Nyaya",
  },
  description,
  applicationName: "Nyaya",
  formatDetection: { telephone: false },
  openGraph: {
    title,
    description,
    siteName: "Nyaya",
    type: "website",
  },
  twitter: {
    card: "summary",
    title,
    description,
  },
  robots: {
    index: false,
    follow: false,
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f4ee" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0c12" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} ${sourceSerif.variable} h-full antialiased`}
    >
      <body className="flex h-full min-h-screen flex-col bg-bg text-text">
        <ThemeProvider attribute="data-theme" defaultTheme="system" enableSystem>
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
