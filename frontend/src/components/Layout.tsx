import { type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AudioWaveform, FlaskConical, Library, Mic, Moon, Settings2, Sun } from "lucide-react";
import { api } from "@/services/api";
import { useAppSettings } from "@/hooks/useAppSettings";
import { cn } from "@/lib/utils";
import { Tip } from "@/components/ui/primitives";

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 64 64" className="h-8 w-8" aria-hidden>
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="rgb(var(--accent))" />
            <stop offset="1" stopColor="rgb(var(--accent2))" />
          </linearGradient>
        </defs>
        <rect width="64" height="64" rx="16" className="fill-elevated" />
        <path d="M10 38c6 0 6-14 12-14s6 22 12 22 6-30 12-30 6 22 8 22" fill="none" stroke="url(#lg)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <span className="text-[17px] font-semibold tracking-tight">
        Melody<span className="gradient-text">AI</span>
      </span>
    </div>
  );
}

function HealthPill() {
  const { data, isError } = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  if (isError)
    return (
      <span className="flex items-center gap-2 text-xs text-bad">
        <span className="h-2 w-2 rounded-full bg-bad" /> API offline
      </span>
    );
  if (!data) return null;
  return (
    <Tip
      content={
        <div className="space-y-1">
          <div>{data.indexed} indexed songs · {data.with_lyrics} with lyrics</div>
          <div>Speech recognition: {data.asr_ready ? "ready" : "not available (type lyrics instead)"}</div>
          <div>Neural model: {data.neural_model ?? "not loaded"}</div>
          <div>Device: {data.device} · ffmpeg: {data.ffmpeg ? "yes" : "no"}</div>
        </div>
      }
    >
      <span className="flex cursor-default items-center gap-2 rounded-full border border-border bg-surface/70 px-3 py-1 text-xs text-muted">
        <span className="h-2 w-2 rounded-full bg-ok shadow-[0_0_8px] shadow-ok" />
        {data.indexed} songs
        {data.asr_ready && <span className="hidden sm:inline">· ASR</span>}
        {data.neural_model && <span className="hidden sm:inline">· neural</span>}
      </span>
    </Tip>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const { t, settings, update } = useAppSettings();
  const nav = [
    { to: "/", label: t("nav.recognize"), icon: Mic },
    { to: "/library", label: t("nav.library"), icon: Library },
    { to: "/explorer", label: t("nav.explorer"), icon: AudioWaveform },
    { to: "/lab", label: t("nav.lab"), icon: FlaskConical },
    { to: "/settings", label: t("nav.settings"), icon: Settings2 },
  ];
  const dark = settings.theme === "dark" || (settings.theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  return (
    <div className="flex min-h-full">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-border/60 bg-surface/40 px-4 py-6 backdrop-blur-xl lg:flex">
        <Logo />
        <nav className="mt-10 space-y-1" aria-label="Main">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn("flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors", isActive ? "bg-elevated text-fg shadow-inner" : "text-muted hover:bg-elevated/60 hover:text-fg")
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto space-y-3 text-xs text-muted">
          <p className="leading-relaxed">Local processing. Recordings are not stored unless you allow it.</p>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-border/50 bg-bg/70 px-4 py-3 backdrop-blur-xl sm:px-8">
          <div className="lg:hidden">
            <Logo />
          </div>
          <div className="hidden lg:block" />
          <div className="flex items-center gap-2">
            <HealthPill />
            <button
              aria-label="Toggle theme"
              onClick={() => update({ theme: dark ? "light" : "dark" })}
              className="rounded-lg p-2 text-muted hover:bg-elevated hover:text-fg"
            >
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            <button
              aria-label="Switch language"
              onClick={() => update({ ui_language: settings.ui_language === "en" ? "es" : "en" })}
              className="rounded-lg px-2 py-1.5 text-xs font-semibold text-muted hover:bg-elevated hover:text-fg"
            >
              {settings.ui_language.toUpperCase()}
            </button>
          </div>
        </header>
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 pb-28 pt-6 sm:px-8 lg:pb-12">
          {children}
        </main>
      </div>
      <nav aria-label="Main mobile" className="fixed inset-x-0 bottom-0 z-30 flex justify-around border-t border-border/60 bg-surface/90 px-2 py-2 backdrop-blur-xl lg:hidden">
        {nav.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => cn("flex flex-col items-center gap-1 rounded-lg px-2 py-1 text-[10px]", isActive ? "text-accent" : "text-muted")}>
            <Icon className="h-5 w-5" />
            {label.split(" ")[0]}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
