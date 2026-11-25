// Splits Markdown into top-level blocks so a streaming reply re-parses only
// its last (still open) block. Conservative: it splits only at a blank line
// that is followed by an unindented line that is not a list item or quote,
// and never inside a code fence. Anything that could join blocks across a
// blank line (link reference definitions, raw HTML) disables splitting.
// Plain TypeScript with no type-only syntax beyond annotations, so the
// node test (tests/bench/md-blocks.mjs) can load it directly.

const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const NO_SPLIT = /^ {0,3}(\[[^\]]+\]:|<)/m;
const LIST_OR_QUOTE = /^(?:[-*+](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$)|>)/;

export function splitBlocks(text: string): string[] {
  if (!text.includes("\n\n") || NO_SPLIT.test(text)) return [text];
  const blocks: string[] = [];
  let start = 0;
  let pos = 0;
  let fenceCh = "";
  let fenceLen = 0;
  let sawBlank = false;
  let sawContent = false;
  while (pos < text.length) {
    let end = text.indexOf("\n", pos);
    const next = end === -1 ? text.length : end + 1;
    if (end === -1) end = text.length;
    const line = text.slice(pos, end);
    const blank = line.trim() === "";
    if (fenceCh) {
      const m = FENCE.exec(line);
      if (m && m[1][0] === fenceCh && m[1].length >= fenceLen && m[2].trim() === "") fenceCh = "";
      sawBlank = false;
    } else if (blank) {
      if (sawContent) sawBlank = true;
    } else {
      if (sawBlank && pos > start && line[0] !== " " && line[0] !== "\t" && !LIST_OR_QUOTE.test(line)) {
        blocks.push(text.slice(start, pos));
        start = pos;
      }
      sawBlank = false;
      sawContent = true;
      const m = FENCE.exec(line);
      if (m && !(m[1][0] === "`" && m[2].includes("`"))) {
        fenceCh = m[1][0];
        fenceLen = m[1].length;
      }
    }
    pos = next;
  }
  blocks.push(text.slice(start));
  return blocks;
}
