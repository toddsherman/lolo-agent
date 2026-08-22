import { copyFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const toolsDirectory = fileURLToPath(new URL(".", import.meta.url));

await build({
  entryPoints: [new URL("visual-editing.source.js", import.meta.url).pathname],
  bundle: true,
  format: "iife",
  minify: true,
  outfile: new URL("../site/visual-editing.js", import.meta.url).pathname,
  target: ["es2020"],
});

await copyFile(
  `${toolsDirectory}node_modules/@sanity/visual-editing-standalone/dist/styles.css`,
  new URL("../site/visual-editing.css", import.meta.url),
);
