import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

// Model output is untrusted: react-markdown escapes raw HTML by default
// (no rehype-raw), so markup in a reply is shown as text, never executed.
// Code blocks are coloured only when they name their language: guessing the
// language (detect) tries every grammar and was the main cost of re-rendering.
export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: false, ignoreMissing: true }]]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
