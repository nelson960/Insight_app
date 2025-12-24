import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ChatWindow } from "./ChatWindow";
import { DocumentsPane } from "./DocumentsPane";
import { SplitView } from "./SplitView";
import type { CardLayout } from "./Canvas";

type Props = {
  title: string;
  chatId: string;
  onClose: () => void;
  initialLayout?: CardLayout;
  onLayoutChange?: (next: CardLayout) => void;
};

const DEFAULT_LAYOUT: CardLayout = {
  showChat: true,
  showDocs: true,
  chatOnRight: true,
  splitRatio: 0.5,
};

export function CardOverlay({ title, chatId, onClose, initialLayout, onLayoutChange }: Props) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showChat, setShowChat] = useState(initialLayout?.showChat ?? DEFAULT_LAYOUT.showChat);
  const [showDocs, setShowDocs] = useState(initialLayout?.showDocs ?? DEFAULT_LAYOUT.showDocs);
  const [chatOnRight, setChatOnRight] = useState(
    initialLayout?.chatOnRight ?? DEFAULT_LAYOUT.chatOnRight
  );
  const [splitRatio, setSplitRatio] = useState(
    initialLayout?.splitRatio ?? DEFAULT_LAYOUT.splitRatio
  );
  const [activeFileId, setActiveFileId] = useState<string | null>(initialLayout?.activeFileId ?? null);
  const [docsRefreshSeq, setDocsRefreshSeq] = useState(0);
  const [selection, setSelection] = useState<{ text: string; file_id?: string } | null>(null);

  const isSplit = showChat && showDocs;

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
    // Ensure at least one pane remains visible.
    if (!showChat && !showDocs) setShowDocs(true);
    onLayoutChange?.({
      showChat,
      showDocs,
      chatOnRight,
      splitRatio,
      activeFileId,
    });
  }, [activeFileId, chatOnRight, onLayoutChange, showChat, showDocs, splitRatio]);

  useEffect(() => {
    // Switching cards/chats should reset which document is selected.
    setActiveFileId(initialLayout?.activeFileId ?? null);
    setSelection(null);
    setDocsRefreshSeq((v) => v + 1);
  }, [chatId, initialLayout?.activeFileId]);

  const docsPane = useMemo(
    () => (
      <DocumentsPane
        chatId={chatId}
        refreshSeq={docsRefreshSeq}
        activeFileId={activeFileId}
        onActiveFileIdChange={setActiveFileId}
        onSelectionChange={setSelection}
        onAskSelection={askSelection}
      />
    ),
    [activeFileId, askSelection, chatId, docsRefreshSeq]
  );

  const chatPane = useMemo(
    () => (
      <ChatWindow
        chatId={chatId}
        active={true}
        embedded={true}
        showTopbar={false}
        activeDocumentId={activeFileId}
        selection={selection}
        onSetSelection={setSelection}
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
    ),
    [activeFileId, chatId, selection]
  );

  const left = useMemo(() => {
    if (isSplit) return chatOnRight ? docsPane : chatPane;
    if (showDocs) return docsPane;
    if (showChat) return chatPane;
    return docsPane;
  }, [chatOnRight, chatPane, docsPane, isSplit, showChat, showDocs]);

  const right = useMemo(() => {
    if (!isSplit) return null;
    return chatOnRight ? chatPane : docsPane;
  }, [chatOnRight, chatPane, docsPane, isSplit]);

  function toggleChatPane() {
    if (showChat) {
      if (!showDocs) {
        // Switching from chat-only → docs-only.
        setShowDocs(true);
      }
      setShowChat(false);
      return;
    }
    setShowChat(true);
  }

  function toggleDocsPane() {
    if (showDocs) {
      if (!showChat) {
        // Switching from docs-only → chat-only.
        setShowChat(true);
      }
      setShowDocs(false);
      return;
    }
    setShowDocs(true);
  }

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
              onClick={toggleChatPane}
              type="button"
              aria-pressed={showChat}
              title={showChat ? "Hide chat" : "Show chat"}
            >
              Chat
            </button>
            <button
              className={`card-overlay-chat-btn ${showDocs ? "active" : ""}`}
              onClick={toggleDocsPane}
              type="button"
              aria-pressed={showDocs}
              title={showDocs ? "Hide documents" : "Show documents"}
            >
              Docs
            </button>
            {isSplit ? (
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
          {isSplit ? (
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
