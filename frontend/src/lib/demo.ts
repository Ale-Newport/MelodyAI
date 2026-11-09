/**
 * Static demo build (GitHub Pages, `VITE_DEMO=1`): there is no server, so API answers come from the JSON files
 * captured by `scripts/build_demo_site.py` into `public/demo/`, and recognition replays example results.
 */
export const DEMO = import.meta.env.VITE_DEMO === "1";

export const REPO_URL = "https://github.com/Ale-Newport/MelodyAI";

/** URL of a file of the captured demo (`public/demo/<path>`), relative to the site's base path. */
export const demoUrl = (path: string) => `${import.meta.env.BASE_URL}demo/${path}`;

export const DEMO_UNAVAILABLE = "Not available in this static demo. Run MelodyAI locally to use it.";
