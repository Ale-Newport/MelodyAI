import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";

export const FINISHED = ["completed", "failed", "cancelled", "interrupted"];

/** Polls a background job until it finishes. */
export function useJob(id: string | null | undefined, onDone?: () => void) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: async () => {
      const j = await api.job(id!);
      if (FINISHED.includes(j.status)) onDone?.();
      return j;
    },
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && FINISHED.includes(q.state.data.status) ? false : 1000),
  });
}
