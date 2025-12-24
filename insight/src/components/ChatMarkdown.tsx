import { memo, useCallback, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";

function safeText(children: unknown) {
  if (Array.isArray(children)) return children.map((c) => safeText(c)).join("");
  return String(children ?? "");
}

async function copyToClipboard(text: string) {
  const cleaned = String(text ?? "");
  if (!cleaned) return;
  try {
    await navigator.clipboard.writeText(cleaned);
  } catch {
    // ignore (best-effort)
  }
}

type MarkdownProps = {
  markdown: string;
};

function MarkdownRenderer({ markdown }: MarkdownProps) {
  const renderCode = useCallback(
    (props: {
      inline?: boolean;
      className?: string;
      children?: unknown;
    }) => {
      const inline = Boolean(props.inline);
      const className = props.className || "";
      const raw = safeText(props.children);
      const text = raw.replace(/\n$/, "");
      const langMatch = /language-([A-Za-z0-9_+-]+)/.exec(className);
      const language = langMatch?.[1] || "";

      if (inline) {
        return <code className="chat-md-code">{props.children as any}</code>;
      }

      return <CodeBlock text={text} language={language} />;
    },
    []
  );

  const components = useMemo(() => {
    return {
      // Unwrap `pre` so our `code` renderer can own block layout without nested <pre>.
      pre: ({ children }: { children?: any }) => <>{children}</>,
      code: renderCode as any,
    };
  }, [renderCode]);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      skipHtml
      components={components}
    >
      {markdown}
    </ReactMarkdown>
  );
}

const MemoMarkdownRenderer = memo(MarkdownRenderer);

function CodeBlock({ text, language }: { text: string; language?: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = useCallback(() => {
    void copyToClipboard(text).finally(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 900);
    });
  }, [text]);

  return (
    <div className="chat-md-codeblock">
      <div className="chat-md-codeblock-toolbar" contentEditable={false}>
        <div className="chat-md-codeblock-left">
          {language ? <span className="chat-md-codeblock-lang">{language}</span> : null}
        </div>
        <button
          type="button"
          className="chat-md-codeblock-copy"
          onClick={onCopy}
          aria-label="Copy code"
          title="Copy"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="chat-md-pre">
        <code className="chat-md-pre-code">{text}</code>
      </pre>
    </div>
  );
}

export const ChatMarkdown = memo(function ChatMarkdown({ markdown }: MarkdownProps) {
  const content = String(markdown ?? "");
  return (
    <div className="chat-md" data-role="assistant-markdown">
      <MemoMarkdownRenderer markdown={content} />
    </div>
  );
});

export const ChatMarkdownStream = memo(function ChatMarkdownStream(props: {
  blocks: string[];
  tail: string;
  inFence?: boolean;
  fenceToken?: "```" | "~~~" | null;
}) {
  const blocks = Array.isArray(props.blocks) ? props.blocks : [];
  const tail = String(props.tail ?? "");
  const inFence = Boolean(props.inFence);
  const fenceToken = props.fenceToken === "~~~" ? "~~~" : "```";
  // During streaming, incomplete fenced code blocks render as plain text until the closing
  // fence arrives. To avoid the "markdown flips at the end" feel, we optimistically close
  // the fence for rendering only (does not change persisted content).
  const tailForRender = inFence && tail ? `${tail}\n${fenceToken}\n` : tail;
  return (
    <div className="chat-md" data-role="assistant-markdown">
      {blocks.map((b, i) => (
        <MemoMarkdownRenderer key={i} markdown={b} />
      ))}
      {tailForRender ? <MemoMarkdownRenderer key="tail" markdown={tailForRender} /> : null}
    </div>
  );
});
