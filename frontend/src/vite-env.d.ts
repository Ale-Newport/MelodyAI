/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" in the static GitHub Pages build (see src/lib/demo.ts). */
  readonly VITE_DEMO?: string;
}

// Font packages imported only for their CSS (TypeScript 7 checks side-effect imports).
declare module "@fontsource-variable/*";
