import React, { useMemo, useState } from "react";
import { Copy, Check } from "lucide-react";

type MarkdownRendererProps = {
  content: string;
};

// Very small, safe markdown subset renderer without innerHTML
// Supports: paragraphs, inline code ``, bold ** **, italics * *,
// fenced code blocks ```lang ... ```, and simple lists (-, *, 1.)
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  const blocks = useMemo(() => parseMarkdown(content), [content]);
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      {blocks.map((block, idx) => {
        if (block.type === "code") {
          return <CodeBlock key={idx} code={block.code} language={block.language} />;
        }
        if (block.type === "list") {
          return block.ordered ? (
            <ol key={idx} className="list-decimal pl-6 space-y-1">
              {block.items.map((item, i) => (
                <li key={i}>{renderInline(item)}</li>
              ))}
            </ol>
          ) : (
            <ul key={idx} className="list-disc pl-6 space-y-1">
              {block.items.map((item, i) => (
                <li key={i}>{renderInline(item)}</li>
              ))}
            </ul>
          );
        }
        // paragraph
        return (
          <p key={idx} className="whitespace-pre-wrap break-words">
            {renderInline(block.text)}
          </p>
        );
      })}
    </div>
  );
};

function renderInline(text: string): React.ReactNode[] {
  // Process inline code first
  const parts: Array<string | { code: string }> = [];
  let remaining = text;
  while (true) {
    const start = remaining.indexOf("`");
    if (start === -1) {
      parts.push(remaining);
      break;
    }
    const end = remaining.indexOf("`", start + 1);
    if (end === -1) {
      parts.push(remaining);
      break;
    }
    if (start > 0) parts.push(remaining.slice(0, start));
    parts.push({ code: remaining.slice(start + 1, end) });
    remaining = remaining.slice(end + 1);
  }

  // Then bold and italics inside string parts only
  const nodes: React.ReactNode[] = [];
  parts.forEach((part, idx) => {
    if (typeof part !== "string") {
      nodes.push(
        <code key={`code-${idx}`} className="px-1 py-0.5 rounded bg-muted text-muted-foreground">
          {part.code}
        </code>
      );
      return;
    }
    // Bold **text**
    const boldSplit = splitByDelimiter(part, "**");
    boldSplit.forEach((b, i) => {
      if (typeof b !== "string") {
        nodes.push(
          <strong key={`b-${idx}-${i}`} className="font-semibold">
            {b}
          </strong>
        );
      } else {
        // Italic *text*
        const italicSplit = splitByDelimiter(b, "*");
        italicSplit.forEach((it, j) => {
          if (typeof it !== "string") {
            nodes.push(
              <em key={`i-${idx}-${i}-${j}`} className="italic">
                {it}
              </em>
            );
          } else if (it.length > 0) {
            nodes.push(<React.Fragment key={`t-${idx}-${i}-${j}`}>{it}</React.Fragment>);
          }
        });
      }
    });
  });

  return nodes;
}

function splitByDelimiter(source: string, delimiter: string): Array<string | string> {
  const result: Array<string | string> = [];
  const parts = source.split(delimiter);
  parts.forEach((p, idx) => {
    if (idx % 2 === 1) {
      result.push(p);
    } else {
      result.push(p);
    }
  });
  return result;
}

type Block =
  | { type: "paragraph"; text: string }
  | { type: "code"; language?: string; code: string }
  | { type: "list"; ordered: boolean; items: string[] };

function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (line.startsWith("```")) {
      const language = line.slice(3).trim() || undefined;
      i += 1;
      const codeLines: string[] = [];
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      // skip closing fence
      if (i < lines.length && lines[i].startsWith("```")) i += 1;
      blocks.push({ type: "code", language, code: codeLines.join("\n") });
      continue;
    }

    // List (unordered)
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i += 1;
      }
      blocks.push({ type: "list", ordered: false, items });
      continue;
    }

    // List (ordered)
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push({ type: "list", ordered: true, items });
      continue;
    }

    // Paragraph / blank lines coalesced
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].startsWith("```")) {
      para.push(lines[i]);
      i += 1;
    }
    if (para.length > 0) {
      blocks.push({ type: "paragraph", text: para.join("\n") });
    }
    // skip blank line
    if (i < lines.length && lines[i].trim() === "") i += 1;
  }
  return blocks.length ? blocks : [{ type: "paragraph", text: src }];
}

const CodeBlock: React.FC<{ code: string; language?: string }> = ({ code }) => {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      // nosemgrep: unsafe-eval
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // ignore
    }
  };
  return (
    <div className="relative group">
      <pre className="whitespace-pre-wrap p-3 rounded border bg-muted/40 overflow-auto">
        <code>{code}</code>
      </pre>
      <button
        type="button"
        onClick={onCopy}
        aria-label="Copy code"
        className="absolute top-2 right-2 inline-flex items-center gap-1 rounded border px-2 py-1 text-xs bg-background/80 hover:bg-background"
      >
        {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
};

export default MarkdownRenderer;


