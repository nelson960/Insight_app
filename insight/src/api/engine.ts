import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

export interface EngineResponse<T = any> {
  ok: boolean;
  status: number;
  data: T;
  error?: string;
}

type StreamDonePayload = { request_id?: string; chat_id?: string };
type StreamErrorPayload = { request_id?: string; chat_id?: string; error?: string };

type ActiveStream = {
  requestId: string;
  chatId: string;
};

let activeStream: ActiveStream | null = null;
const finishedRequestIds = new Map<string, number>(); // requestId -> expiresAt (ms)
const finishWaiters = new Map<
  string,
  { resolve: () => void; reject: (err: Error) => void; timer?: number }
>();
let streamBridgeInit: Promise<void> | null = null;
let unlistenDone: UnlistenFn | null = null;
let unlistenError: UnlistenFn | null = null;

async function ensureStreamBridge() {
  if (streamBridgeInit) return streamBridgeInit;
  streamBridgeInit = (async () => {
    unlistenDone = await listen<StreamDonePayload>("llm-done", (event) => {
      const rid = event.payload?.request_id;
      if (!rid) return;
      if (activeStream?.requestId === rid) activeStream = null;
      finishedRequestIds.set(rid, Date.now() + 60_000);
      const waiter = finishWaiters.get(rid);
      if (waiter) {
        finishWaiters.delete(rid);
        if (waiter.timer) window.clearTimeout(waiter.timer);
        waiter.resolve();
      }
    });

    unlistenError = await listen<StreamErrorPayload>("llm-error", (event) => {
      const rid = event.payload?.request_id;
      if (!rid) return;
      // Important: don't clear `activeStream` on llm-error. Rust emits llm-error as soon
      // as it sees a stream_error line, but the stream thread is still consuming stdout
      // until it sees stream_end and emits llm-done. Clearing early can allow a new stream
      // to start and cause stdout contention (dropped tokens / stuck streams).
    });
  })();
  return streamBridgeInit;
}

export function getActiveStream() {
  return activeStream;
}

export async function waitForStreamToFinish(requestId: string, timeoutMs = 45_000) {
  await ensureStreamBridge();
  if (!requestId) return;
  const expiresAt = finishedRequestIds.get(requestId);
  if (expiresAt && expiresAt > Date.now()) return;
  if (!finishWaiters.has(requestId)) {
    let timer: number | undefined;
    const p = new Promise<void>((resolve, reject) => {
      timer = window.setTimeout(() => {
        finishWaiters.delete(requestId);
        reject(new Error("timeout waiting for stream to finish"));
      }, timeoutMs);
      finishWaiters.set(requestId, { resolve, reject, timer });
    });
    return p;
  }
  // A wait is already registered; return a promise that resolves when it does.
  return new Promise<void>((resolve, reject) => {
    const existing = finishWaiters.get(requestId);
    if (!existing) return resolve();
    const prevResolve = existing.resolve;
    const prevReject = existing.reject;
    existing.resolve = () => {
      prevResolve();
      resolve();
    };
    existing.reject = (e) => {
      prevReject(e);
      reject(e);
    };
  });
}

export async function engine<T = any>(
  endpoint: string,
  payload?: any,
  method = "POST"
): Promise<EngineResponse<T>> {
  try {
    const invokeArgs: Record<string, any> = {
      endpoint,
      method,
    };
    if (typeof payload !== "undefined") {
      invokeArgs.payload = payload;
    }
    const res = await invoke<EngineResponse<T>>("engine_request", {
      ...invokeArgs,
    });
    return res;
  } catch (err: any) {
    return {
      ok: false,
      status: 500,
      data: null as T,
      error: err?.message ?? "Unknown engine error",
    };
  }
}

/**
 * Streaming helper for chat. Ensures we always set stream:true so the Python
 * engine stays in streaming mode end-to-end.
 */
export async function engineStreamChat(opts: {
  chatId: string;
  query: string;
  requestId: string;
  paths?: string[];
  documents?: string[];
  focusDocumentId?: string | null;
  selection?: { text: string; file_id?: string; page?: number } | null;
}) {
  const { chatId, query, requestId, paths, documents, focusDocumentId, selection } = opts;
  await ensureStreamBridge();
  activeStream = { requestId, chatId };
  return invoke("engine_stream_request", {
    requestId,
    payload: {
      chat_id: chatId,
      query,
      stream: true,
      ...(focusDocumentId ? { focus_document_id: focusDocumentId } : {}),
      ...(selection && selection.text ? { selection } : {}),
      ...(documents && documents.length ? { documents } : {}),
      ...(paths && paths.length ? { paths } : {}),
    },
  });
}

export async function engineCancel(requestId: string) {
  return invoke("engine_cancel_request", { requestId });
}

export async function cancelActiveStreamAndWait(timeoutMs = 45_000) {
  const current = activeStream;
  if (!current) return;
  try {
    await engineCancel(current.requestId);
  } catch {
    // ignore
  }
  await waitForStreamToFinish(current.requestId, timeoutMs);
}
