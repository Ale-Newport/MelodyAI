import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";
import App from "./App";
import { SettingsProvider } from "@/hooks/useAppSettings";
import { TooltipProvider } from "@/components/ui/primitives";
import { DEMO } from "@/lib/demo";

// GitHub Pages serves only index.html, so the static demo keeps the route in the URL hash.
const Router = DEMO ? HashRouter : BrowserRouter;

const qc = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } } });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <Router>
        <SettingsProvider>
          <TooltipProvider>
            <App />
          </TooltipProvider>
        </SettingsProvider>
      </Router>
    </QueryClientProvider>
  </React.StrictMode>,
);
