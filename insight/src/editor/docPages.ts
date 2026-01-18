import { engine, EngineResponse } from "../api/engine";

export type DocPageResponse = {
  chat_id: string;
  file_id: string;
  title: string;
  doc: any;
  updated_at?: string | null;
  source_file_updated_at?: string | null;
  source_is_stale?: boolean;
  is_user_edited?: boolean;
  bootstrapped?: boolean;
};

// Simple in-memory cache for document pages
const docPageCache = new Map<string, { data: DocPageResponse; timestamp: number }>();
const CACHE_TTL_MS = 60000; // 1 minute cache

function getCacheKey(chatId: string, fileId: string): string {
  return `${chatId}:${fileId}`;
}

function getCached(chatId: string, fileId: string): DocPageResponse | null {
  const key = getCacheKey(chatId, fileId);
  const entry = docPageCache.get(key);
  if (!entry) return null;

  // Check if cache is still valid
  const now = Date.now();
  if (now - entry.timestamp > CACHE_TTL_MS) {
    docPageCache.delete(key);
    return null;
  }

  return entry.data;
}

function setCached(chatId: string, fileId: string, data: DocPageResponse): void {
  const key = getCacheKey(chatId, fileId);
  docPageCache.set(key, { data, timestamp: Date.now() });

  // Limit cache size (keep most recent 20 entries)
  if (docPageCache.size > 20) {
    const entries = Array.from(docPageCache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
    // Remove oldest 5 entries
    for (let i = 0; i < 5 && i < entries.length; i++) {
      docPageCache.delete(entries[i][0]);
    }
  }
}

function invalidateCache(chatId: string, fileId: string): void {
  const key = getCacheKey(chatId, fileId);
  docPageCache.delete(key);
}

export async function loadDocPage(chatId: string, fileId: string, forceReload = false): Promise<EngineResponse<DocPageResponse>> {
  // Check cache first (unless force reload)
  if (!forceReload) {
    const cached = getCached(chatId, fileId);
    if (cached) {
      return { ok: true, status: 200, data: cached };
    }
  }

  const result = await engine<DocPageResponse>(`/docs/page/${encodeURIComponent(chatId)}/${encodeURIComponent(fileId)}`, undefined, "GET");

  // Cache successful responses
  if (result.ok && result.data) {
    setCached(chatId, fileId, result.data);
  }

  return result;
}

export async function saveDocPage(chatId: string, fileId: string, opts: { title?: string; doc: any }) {
  const payload: any = { doc: opts.doc };
  if (typeof opts.title === "string") payload.title = opts.title;

  const result = await engine(`/docs/page/${encodeURIComponent(chatId)}/${encodeURIComponent(fileId)}`, payload, "PUT");

  // Invalidate cache on successful save
  if (result.ok) {
    invalidateCache(chatId, fileId);
  }

  return result;
}

export function clearDocPageCache(): void {
  docPageCache.clear();
}

