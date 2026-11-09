import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Trash2 } from "lucide-react";
import { api } from "@/services/api";
import { ResultsView } from "@/features/recognition/ResultsView";
import { Button, ErrorBox, Skeleton } from "@/components/ui/primitives";
import { DEMO } from "@/lib/demo";

export default function RecognitionPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["recognition", id], queryFn: () => api.recognition(id) });
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Link to="/" className="inline-flex items-center gap-1 text-sm text-muted hover:text-fg">
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>
        {!DEMO && <Button
          size="sm"
          variant="danger"
          onClick={async () => {
            await api.deleteRecognition(id);
            nav("/");
          }}
        >
          <Trash2 className="h-4 w-4" /> Delete from history
        </Button>}
      </div>
      {q.isLoading && <Skeleton className="h-64" />}
      {q.isError && <ErrorBox error={q.error} />}
      {q.data && <ResultsView result={q.data} onAgain={() => nav("/")} />}
    </div>
  );
}
