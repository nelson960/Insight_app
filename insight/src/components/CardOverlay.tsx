import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ChatWindow } from "./ChatWindow";
import { DocumentsPane } from "./DocumentsPane";
import { SplitView } from "./SplitView";

type Props = {
  title: string;
  chatId: string;
  onClose: () => void;
  initialShowChat?: boolean;
};

export function CardOverlay({ title, chatId, onClose, initialShowChat = true }: Props) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showChat, setShowChat] = useState(initialShowChat);
  const [chatOnRight, setChatOnRight] = useState(true);
  const [splitRatio, setSplitRatio] = useState(0.5);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [docsRefreshSeq, setDocsRefreshSeq] = useState(0);
  const [selection, setSelection] = useState<{ file_id: string; text: string } | null>(null);

  const askSelection = useCallback(() => {
    setShowChat(true);
    // After chat is shown, focus the input.
    window.setTimeout(() => {
      try {
        window.dispatchEvent(new CustomEvent("insight:focus-chat", { detail: { chatId } }));
      } catch {
        // ignore
      }
    }, 0);
  }, [chatId]);

  useEffect(() => {
    setShowChat(initialShowChat);
  }, [initialShowChat]);

  useEffect(() => {
    // Switching cards/chats should reset which document is selected.
    setActiveFileId(null);
    setSelection(null);
    setDocsRefreshSeq((v) => v + 1);
  }, [chatId]);

  const left = useMemo(() => {
    const docsPane = (
      <DocumentsPane
        chatId={chatId}
        refreshSeq={docsRefreshSeq}
        activeFileId={activeFileId}
        onActiveFileIdChange={setActiveFileId}
        onSelectionChange={setSelection}
        onAskSelection={askSelection}
      />
    );
    const chatPane = (
      <ChatWindow
        chatId={chatId}
        active={true}
        embedded={true}
        showTopbar={false}
        activeDocumentId={activeFileId}
        selection={selection}
        onClearSelection={() => {
          setSelection(null);
          try {
            window.getSelection?.()?.removeAllRanges?.();
          } catch {
            // ignore
          }
        }}
        onRequestDocsRefresh={() => setDocsRefreshSeq((v) => v + 1)}
      />
    );
    if (!showChat) return docsPane;
    return chatOnRight ? docsPane : chatPane;
  }, [activeFileId, chatId, chatOnRight, docsRefreshSeq, showChat, selection]);

  const right = useMemo(() => {
    const docsPane = (
      <DocumentsPane
        chatId={chatId}
        refreshSeq={docsRefreshSeq}
        activeFileId={activeFileId}
        onActiveFileIdChange={setActiveFileId}
        onSelectionChange={setSelection}
        onAskSelection={askSelection}
      />
    );
    const chatPane = (
      <ChatWindow
        chatId={chatId}
        active={true}
        embedded={true}
        showTopbar={false}
        activeDocumentId={activeFileId}
        selection={selection}
        onClearSelection={() => {
          setSelection(null);
          try {
            window.getSelection?.()?.removeAllRanges?.();
          } catch {
            // ignore
          }
        }}
        onRequestDocsRefresh={() => setDocsRefreshSeq((v) => v + 1)}
      />
    );
    if (!showChat) return null;
    return chatOnRight ? chatPane : docsPane;
  }, [activeFileId, chatId, chatOnRight, docsRefreshSeq, showChat, selection]);

  return (
    <div
      className={`chat-overlay card-overlay ${isFullscreen ? "card-overlay-fullscreen" : ""}`}
      style={{ background: "rgba(0, 0, 0, 0.55)" }}
      role="dialog"
      aria-modal="true"
      aria-label="Card"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={`chat-overlay-window ${isFullscreen ? "card-overlay-window-fullscreen" : ""}`}
        style={{ background: "#0f172a" }}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div
          className="chat-overlay-header"
          onDoubleClick={() => setIsFullscreen((v) => !v)}
          title="Double-click to toggle full screen"
        >
          <div className="chat-overlay-title">{title}</div>
          <div className="card-overlay-actions">
            <button
              className="card-overlay-full-btn"
              onClick={() => setIsFullscreen((v) => !v)}
              aria-label={isFullscreen ? "Exit full screen" : "Full screen"}
              title={isFullscreen ? "Exit full screen" : "Full screen"}
              type="button"
            >
              {isFullscreen ? "⤡" : "⤢"}
            </button>
            <button
              className={`card-overlay-chat-btn ${showChat ? "active" : ""}`}
              onClick={() => setShowChat((v) => !v)}
              type="button"
              aria-pressed={showChat}
              title={showChat ? "Hide chat" : "Show chat"}
            >
              {showChat ? "Docs" : "Chat"}
            </button>
            {showChat ? (
              <button
                className="card-overlay-swap-btn"
                onClick={() => setChatOnRight((v) => !v)}
                type="button"
                title="Swap panes"
                aria-label="Swap panes"
              >
                ⇄
              </button>
            ) : null}
            <button className="chat-overlay-close" onClick={onClose}>
              ×
            </button>
          </div>
        </div>
        <div className="card-overlay-body">
          {showChat ? (
            <SplitView
              left={left}
              right={right}
              ratio={splitRatio}
              onChangeRatio={setSplitRatio}
              minLeftPx={280}
              minRightPx={320}
              dividerPx={12}
            />
          ) : (
            left
          )}
        </div>
      </div>
    </div>
  );
}
