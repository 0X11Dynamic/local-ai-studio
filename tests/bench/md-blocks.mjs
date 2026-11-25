// Correctness check for the streaming Markdown block splitter (frontend/src/lib/mdBlocks.ts).
// For each sample, and for every prefix of it (what a streaming reply looks like), the blocks
// must concatenate back to the input and render to the same HTML as a full parse.
//   node tests/bench/md-blocks.mjs
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL, fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fe = path.join(root, "frontend");
const require = createRequire(path.join(fe, "package.json"));
const imp = (m) => import(pathToFileURL(require.resolve(m)).href);
const React = await imp("react");
const { renderToStaticMarkup } = await imp("react-dom/server");
const RM = (await imp("react-markdown")).default;
const gfm = (await imp("remark-gfm")).default;
const hl = (await imp("rehype-highlight")).default;
// Node strips the TS types natively (Node 22.18+ / 23.6+).
const { splitBlocks } = await import(pathToFileURL(path.join(fe, "src/lib/mdBlocks.ts")).href);

// react-markdown emits newline text nodes between block elements; they have no visual effect, so ignore them.
const norm = (h) => h.replace(/>\s+</g, "><");
const render = (t) => norm(renderToStaticMarkup(React.createElement(RM, { remarkPlugins: [gfm], rehypePlugins: [[hl, { detect: false, ignoreMissing: true }]] }, t)));
const renderBlocks = (t) => splitBlocks(t).map(render).join("");

const samples = {
  paragraphs: "First para.\n\nSecond para with **bold**.\n\nThird.\n",
  fenceWithBlanks: "Intro\n\n```js\nconst a = 1;\n\n\nconst b = 2;\n```\n\nAfter the code.\n\n~~~py\nx = 1\n\ny = 2\n~~~\n\nEnd\n",
  fenceLonger: "Text\n\n````md\n```\ninner\n\nmore\n```\n````\n\nDone\n",
  nestedLists: "Items:\n\n- one\n  - nested a\n\n  - nested b\n\n- two\n\n  continued para\n\n1. first\n\n2. second\n   - sub\n\nTail paragraph\n",
  table: "Before\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nAfter table\n\n| x |\n|---|\n| y |\n",
  headingsQuotes: "# Title\n\nPara\n\n> quote line\n> more\n\n> second quote\n\n## Sub\n\n---\n\nEnd\n",
  linkRefs: "See [a][1] and more.\n\nSecond.\n\n[1]: https://example.com\n",
  unclosedFence: "Para\n\n```rust\nfn main() {\n\n    println!(\"hi\");\n",
  indentedCode: "Para\n\n    code line 1\n\n    code line 2\n\nEnd\n",
  htmlish: "Text\n\n<div>\nraw\n\nmore</div>\n\nEnd\n",
};

let fail = 0;
let checks = 0;
for (const [name, text] of Object.entries(samples)) {
  for (let n = 1; n <= text.length; n++) {
    const t = text.slice(0, n);
    const blocks = splitBlocks(t);
    checks++;
    if (blocks.join("") !== t) { fail++; console.log(`FAIL concat ${name}@${n}`); break; }
    if (renderBlocks(t) !== render(t)) { fail++; console.log(`FAIL render ${name}@${n}: ${JSON.stringify(t.slice(-40))}`); break; }
  }
}
const multi = samples.fenceWithBlanks;
const nb = splitBlocks(multi).length;
if (nb < 4) { fail++; console.log(`FAIL split count (expected >= 4 blocks, got ${nb})`); }
console.log(`${checks} prefix checks, ${fail} failures`);
process.exit(fail ? 1 : 0);
