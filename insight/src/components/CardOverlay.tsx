import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileText, MessageSquare, Settings, ArrowLeftRight, X } from "lucide-react";
import NewCardIcon from "./icons/NewCardIcon";
import { ChatWindow } from "./ChatWindow";
import { DocumentsPane } from "./DocumentsPane";
import { SplitView } from "./SplitView";
import type { CardLayout } from "./Canvas";
import type { ChatSummary } from "../state/useSessions";

type Props = {
  title: string;
  chatId: string;
  onClose: () => void;
  initialLayout?: CardLayout;
  onLayoutChange?: (next: CardLayout) => void;
  onTitleChange?: (newTitle: string) => void;
  sessions: ChatSummary[];
  loadingSessions: boolean;
  onOpenCard: (chatId: string) => void;
  onCreateCard: () => string;
  onOpenSettings: () => void;
  onDeleteChat: (chatId: string) => void;
  confirmDeleteChatId: string | null;
  onRequireModel?: () => Promise<boolean>;
  closing?: boolean;
};

const DEFAULT_LAYOUT: CardLayout = {
  showChat: true,
  showDocs: true,
  chatOnRight: true,
  splitRatio: 0.5,
};

export function CardOverlay({
  title,
  chatId,
  onClose,
  initialLayout,
  onLayoutChange,
  onTitleChange,
  sessions,
  loadingSessions,
  onOpenCard,
  onCreateCard,
  onOpenSettings,
  onDeleteChat,
  confirmDeleteChatId,
  onRequireModel,
  closing = false,
}: Props) {
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
  const [cardsOpen, setCardsOpen] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [titleDraft, setTitleDraft] = useState(title);
  const cardsWrapRef = useRef<HTMLDivElement | null>(null);
  const cardsMenuRef = useRef<HTMLDivElement | null>(null);
  const renameInputRef = useRef<HTMLInputElement | null>(null);

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

  useEffect(() => {
    // Avoid stale selection carrying across document switches (can bias retrieval).
    setSelection(null);
  }, [activeFileId]);

  useEffect(() => {
    if (!cardsOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as Node | null;
      if (!t) return;
      if (cardsMenuRef.current && cardsMenuRef.current.contains(t)) return;
      if (cardsWrapRef.current && cardsWrapRef.current.contains(t)) return;
      setCardsOpen(false);
    };
    window.addEventListener("pointerdown", onPointerDown, { capture: true });
    return () => window.removeEventListener("pointerdown", onPointerDown, { capture: true } as any);
  }, [cardsOpen]);

  // Sync titleDraft when title prop changes (but not during renaming)
  useEffect(() => {
    if (!isRenaming) {
      setTitleDraft(title);
    }
  }, [title, isRenaming]);

  // Focus the rename input when renaming starts
  useEffect(() => {
    if (isRenaming) {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }
  }, [isRenaming]);

  const docsPane = useMemo(
    () => (
      <DocumentsPane
        chatId={chatId}
        refreshSeq={docsRefreshSeq}
        activeFileId={activeFileId}
        onActiveFileIdChange={setActiveFileId}
        onSelectionChange={setSelection}
        onAskSelection={askSelection}
        onRequireModel={onRequireModel}
      />
    ),
    [activeFileId, askSelection, chatId, docsRefreshSeq, onRequireModel]
  );

  const chatPane = useMemo(
    () => (
      <ChatWindow
        chatId={chatId}
        active={true}
        embedded={true}
        showTopbar={false}
        activeDocumentId={activeFileId}
        docsVisible={showDocs}
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
        onRequireModel={onRequireModel}
      />
    ),
    [activeFileId, chatId, selection, showDocs, onRequireModel]
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

  function commitRename(nextTitle: string) {
    const trimmed = (nextTitle || "").trim();
    onTitleChange?.(trimmed ? trimmed : title);
    setIsRenaming(false);
  }

  function cancelRename() {
    setTitleDraft(title);
    setIsRenaming(false);
  }

  return (
    <div
      className="chat-overlay card-overlay card-overlay-fullscreen"
      data-state={closing ? "closing" : "open"}
      role="dialog"
      aria-modal="true"
      aria-label="Card"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="chat-overlay-window"
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div className="chat-overlay-header card-overlay-header">
          <div className="card-overlay-header-left">
            {chatOnRight ? (
              <button
                className={`card-overlay-docs-btn ${showDocs ? "active" : ""}`}
                type="button"
                aria-label={showDocs ? "Hide documents" : "Show documents"}
                title={showDocs ? "Hide documents" : "Documents"}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  toggleDocsPane();
                }}
              >
                <FileText className="w-4 h-4" />
              </button>
            ) : (
              <button
                className={`card-overlay-chat-btn ${showChat ? "active" : ""}`}
                onClick={toggleChatPane}
                type="button"
                aria-pressed={showChat}
                title={showChat ? "Hide chat" : "Show chat"}
              >
                <MessageSquare className="w-4 h-4" />
              </button>
            )}
          </div>
          {isRenaming ? (
            <input
              ref={renameInputRef}
              className="card-overlay-title-input"
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.stopPropagation();
                  commitRename(titleDraft);
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  e.stopPropagation();
                  cancelRename();
                }
              }}
              onBlur={() => commitRename(titleDraft)}
              aria-label="Rename card"
            />
          ) : (
            <div
              className="chat-overlay-title card-overlay-title"
              title={title}
              onDoubleClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setTitleDraft(title);
                setIsRenaming(true);
              }}
            >
              {title}
            </div>
          )}
          <div className="card-overlay-header-right">
            <div className="card-overlay-actions">
            {chatOnRight ? (
              <button
                className={`card-overlay-chat-btn ${showChat ? "active" : ""}`}
                onClick={toggleChatPane}
                type="button"
                aria-pressed={showChat}
                title={showChat ? "Hide chat" : "Show chat"}
              >
                <MessageSquare className="w-4 h-4" />
              </button>
            ) : (
              <button
                className={`card-overlay-docs-btn ${showDocs ? "active" : ""}`}
                type="button"
                aria-label={showDocs ? "Hide documents" : "Show documents"}
                title={showDocs ? "Hide documents" : "Documents"}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  toggleDocsPane();
                }}
              >
                <FileText className="w-4 h-4" />
              </button>
            )}
            <div className="card-overlay-cards-wrap" ref={cardsWrapRef}>
            <button
              className={`card-overlay-cards-btn ${cardsOpen ? "active" : ""}`}
              type="button"
              aria-label={cardsOpen ? "Close cards" : "Open cards"}
              title={cardsOpen ? "Close cards" : "Cards"}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setCardsOpen((v) => !v);
              }}
            >
              ⋯
            </button>
            {cardsOpen ? (
              <div className="card-overlay-cards-menu" ref={cardsMenuRef}>
                <div className="card-overlay-cards-menu-row">
                  <button
                    className="canvas-dock-icon card-menu-btn"
                    type="button"
                    aria-label="Settings"
                    title="Settings"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setCardsOpen(false);
                      onOpenSettings();
                    }}
                  >
                    <Settings className="w-4 h-4" />
                  </button>
                  {isSplit ? (
                    <button
                      className="canvas-dock-icon card-menu-btn"
                      type="button"
                      aria-label="Swap panes"
                      title="Swap panes"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setChatOnRight((v) => !v);
                      }}
                    >
                      <ArrowLeftRight className="w-4 h-4" />
                    </button>
                  ) : null}
                  <button
                    className="canvas-dock-btn card-menu-btn btn-large-icon"
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      const created = onCreateCard();
                      setCardsOpen(false);
                      if (created) onOpenCard(created);
                    }}
                    title="Create new card"
                  >
                    <NewCardIcon className="w-7 h-7" />
                  </button>
                </div>
                <div className="canvas-chatlist" role="menu" aria-label="Cards">
                  {loadingSessions ? <div className="canvas-chatlist-muted">Syncing…</div> : null}
                  {!loadingSessions && sessions.length === 0 ? (
                    <div className="canvas-chatlist-muted">No cards yet</div>
                  ) : null}
                  {sessions.map((s) => (
                    <div key={s.chat_id} className="canvas-chatlist-row">
                      <button
                        type="button"
                        className={`canvas-chatlist-item ${s.chat_id === chatId ? "active" : ""}`}
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          setCardsOpen(false);
                          onOpenCard(s.chat_id);
                        }}
                        title={s.chat_id}
                      >
                        <span className="canvas-chatlist-item-text">{s.title || s.chat_id}</span>
                        <button
                          type="button"
                          className={`canvas-chatlist-del ${confirmDeleteChatId === s.chat_id ? "confirm" : ""}`}
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            onDeleteChat(s.chat_id);
                          }}
                          title={
                            confirmDeleteChatId === s.chat_id
                              ? "Click again to confirm delete"
                              : "Delete card"
                          }
                          aria-label={`Delete card ${s.chat_id}`}
                        >
                          {confirmDeleteChatId === s.chat_id ? "Del" : <X className="w-3 h-3" />}
                        </button>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            </div>
            <button className="chat-overlay-close" onClick={onClose}>
              ×
            </button>
            </div>
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
