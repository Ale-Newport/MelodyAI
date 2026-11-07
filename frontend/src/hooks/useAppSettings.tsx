import { createContext, useCallback, useContext, useEffect, useMemo, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import type { UserSettings } from "@/types/api";
import { translate, type Lang, type TKey } from "@/lib/i18n";

const DEFAULTS: UserSettings = {
  ui_language: "en",
  theme: "dark",
  default_mode: "auto",
  melody_algorithm: "ensemble",
  pitch_extractor: "pyin",
  fusion_method: "weighted",
  asr_enabled: true,
  asr_model: "base",
  asr_language: null,
  inference_device: "auto",
  min_record_seconds: 5,
  max_record_seconds: 20,
  top_k: 5,
  store_history: true,
  retention: "discard",
  microphone_id: null,
};

interface Ctx {
  settings: UserSettings;
  loaded: boolean;
  update: (patch: Partial<UserSettings>) => Promise<UserSettings>;
  t: (key: TKey, vars?: Record<string, string | number>) => string;
}

const SettingsContext = createContext<Ctx | null>(null);

function readLocal(): Partial<UserSettings> {
  try {
    return JSON.parse(localStorage.getItem("melodyai.settings") ?? "{}");
  } catch {
    return {};
  }
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["settings"], queryFn: api.settings, staleTime: Infinity, retry: 1 });
  const local = useMemo(readLocal, []);
  // The microphone is a per-device choice: keep it in this browser only.
  const settings: UserSettings = { ...DEFAULTS, ...(q.data ?? {}), microphone_id: local.microphone_id ?? null, ...(q.data ? {} : local) };

  const m = useMutation({
    mutationFn: (patch: Partial<UserSettings>) => api.saveSettings(patch),
    onSuccess: (s) => qc.setQueryData(["settings"], s),
  });

  const update = useCallback(
    async (patch: Partial<UserSettings>) => {
      if ("microphone_id" in patch) {
        try {
          localStorage.setItem("melodyai.settings", JSON.stringify({ ...readLocal(), microphone_id: patch.microphone_id }));
        } catch {
          /* storage unavailable */
        }
      }
      return m.mutateAsync(patch);
    },
    [m],
  );

  useEffect(() => {
    const root = document.documentElement;
    const apply = () => {
      const dark = settings.theme === "dark" || (settings.theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
      root.classList.toggle("dark", dark);
    };
    apply();
    const mq = matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener?.("change", apply);
    root.lang = settings.ui_language;
    return () => mq.removeEventListener?.("change", apply);
  }, [settings.theme, settings.ui_language]);

  const t = useCallback(
    (key: TKey, vars?: Record<string, string | number>) => translate(settings.ui_language as Lang, key, vars),
    [settings.ui_language],
  );

  return <SettingsContext.Provider value={{ settings, loaded: q.isSuccess, update, t }}>{children}</SettingsContext.Provider>;
}

export function useAppSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useAppSettings outside provider");
  return ctx;
}
