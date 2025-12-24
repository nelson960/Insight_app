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

export async function loadDocPage(chatId: string, fileId: string): Promise<EngineResponse<DocPageResponse>> {
  return engine<DocPageResponse>(`/docs/page/${encodeURIComponent(chatId)}/${encodeURIComponent(fileId)}`, undefined, "GET");
}

export async function saveDocPage(chatId: string, fileId: string, opts: { title?: string; doc: any }) {
  const payload: any = { doc: opts.doc };
  if (typeof opts.title === "string") payload.title = opts.title;
  return engine(`/docs/page/${encodeURIComponent(chatId)}/${encodeURIComponent(fileId)}`, payload, "PUT");
}

