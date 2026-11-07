import { Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import RecognizePage from "@/pages/RecognizePage";
import RecognitionPage from "@/pages/RecognitionPage";
import LibraryPage from "@/pages/LibraryPage";
import ExplorerPage from "@/pages/ExplorerPage";
import LabPage from "@/pages/LabPage";
import SettingsPage from "@/pages/SettingsPage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<RecognizePage />} />
        <Route path="/r/:id" element={<RecognitionPage />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="/explorer" element={<ExplorerPage />} />
        <Route path="/lab" element={<LabPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<div className="p-10 text-muted">Page not found.</div>} />
      </Routes>
    </Layout>
  );
}
