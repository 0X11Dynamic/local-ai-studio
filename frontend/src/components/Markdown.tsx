import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { splitBlocks } from "../lib/mdBlocks";

// Model output is untrusted: react-markdown escapes raw HTML by default
// (no rehype-raw), so markup in a reply is shown as text, never executed.
// Code blocks are coloured only when they name their language: guessing the
// language (detect) tries every grammar and was the main cost of re-rendering.
const REMARK = [remarkGfm];
const REHYPE: [typeof rehypeHighlight, { detect: boolean; ignoreMissing: boolean }][] = [[rehypeHighlight, { detect: false, ignoreMissing: true }]];
const COMPONENTS = {
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
    <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>
  ),
};

const Body = memo(function Body({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={REMARK} rehypePlugins={REHYPE} components={COMPONENTS}>
      {text}
    </ReactMarkdown>
  );
});

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <Body text={text} />
    </div>
  );
});

/** For a reply that is still streaming: finished blocks are memoized by text, so each update parses only the last open block. */
export const StreamingMarkdown = memo(function StreamingMarkdown({ text }: { text: string }) {
  const blocks = useMemo(() => splitBlocks(text), [text]);
  return (
    <div className="md">
      {blocks.map((b, i) => <Body key={i} text={b} />)}
    </div>
  );
});
