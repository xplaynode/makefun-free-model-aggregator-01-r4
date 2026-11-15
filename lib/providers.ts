// providers.ts — talking to the trial env: the instance directory,
// one catalog probe per provider, and inference calls on each
// provider's own wire with its own auth scheme.
//
// Catalog probes are GETs on the provider's models route. They happen
// once, before serving, and the env does not count them as provider
// calls. Inference calls are always non-streaming: the complete answer
// is translated and, when the client asked for a stream, re-emitted as
// a synthesized event stream in the client's wire. That keeps one code
// path for both shapes and makes the served bytes a pure function of
// the provider's answer.

import {
  buildAnthropicRequest,
  buildGeminiRequest,
  buildOpenAiRequest,
  isRecord,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAiAnswer,
  type IrAnswer,
  type IrRequest,
} from "./wire.ts";
import type { WorkloadItem } from "./plan.ts";

export interface ProviderInfo {
  id: string;
  name: string;
  wire: string; // openai | anthropic | gemini
  auth: string; // bearer | x-api-key | x-goog-api-key | none
  keyEnv: string;
  key: string;
  path: string; // the provider's own path, e.g. /p/groq (catalog form)
  baseUrl: string; // absolute URL to dial, e.g. http://env/p/groq
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface ProbeResult {
  ok: boolean;
  models: CatalogModel[];
}

export type CallOutcome =
  | { ok: true; answer: IrAnswer }
  | { ok: false; status: number; code: string | null };

const ANTHROPIC_VERSION = "2023-06-01";

export function authHeaders(provider: ProviderInfo): Record<string, string> {
  switch (provider.auth) {
    case "bearer":
      return { authorization: `Bearer ${provider.key}` };
    case "x-api-key":
      return { "x-api-key": provider.key };
    case "x-goog-api-key":
      return { "x-goog-api-key": provider.key };
    default:
      return {};
  }
}

function modelsUrl(provider: ProviderInfo): string {
  if (provider.wire === "gemini") return `${provider.baseUrl}/v1beta/models`;
  return `${provider.baseUrl}/v1/models`;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function catalogModels(wire: string, json: Record<string, unknown>): CatalogModel[] {
  const out: CatalogModel[] = [];
  const rows: unknown[] =
    wire === "gemini"
      ? Array.isArray(json.models)
        ? json.models
        : []
      : Array.isArray(json.data)
        ? json.data
        : [];
  for (const row of rows) {
    if (!isRecord(row)) continue;
    let id: string | null = null;
    if (wire === "gemini") {
      if (typeof row.name === "string") id = row.name.replace(/^models\//, "");
    } else if (typeof row.id === "string") {
      id = row.id;
    }
    if (id === null || id === "") continue;
    out.push({
      id,
      free: row.free === true,
      context_length: numberOrNull(row.context_length) ?? numberOrNull(row.inputTokenLimit),
    });
  }
  return out;
}

// One catalog probe. Any failure (route down, bad auth, unparseable
// body) marks the provider not-ok; the caller then gives it a plan
// cap of zero and builds its catalog entry from the brief instead.
export async function probeProvider(provider: ProviderInfo): Promise<ProbeResult> {
  try {
    const headers: Record<string, string> = { ...authHeaders(provider) };
    if (provider.wire === "anthropic") headers["anthropic-version"] = ANTHROPIC_VERSION;
    const response = await fetch(modelsUrl(provider), { headers });
    if (response.status < 200 || response.status >= 300) {
      await response.text().catch(() => "");
      return { ok: false, models: [] };
    }
    const json: unknown = await response.json().catch(() => null);
    if (!isRecord(json)) return { ok: false, models: [] };
    const models = catalogModels(provider.wire, json);
    if (models.length === 0) return { ok: false, models: [] };
    return { ok: true, models };
  } catch {
    return { ok: false, models: [] };
  }
}

export interface InstanceData {
  providers: ProviderInfo[];
  manifest: WorkloadItem[];
}

export async function fetchInstance(envUrl: string): Promise<InstanceData> {
  const response = await fetch(`${envUrl}/instance`);
  if (response.status < 200 || response.status >= 300) {
    throw new Error(`GET /instance answered ${response.status}`);
  }
  const json: unknown = await response.json();
  if (!isRecord(json) || !Array.isArray(json.providers)) {
    throw new Error("GET /instance answered no provider directory");
  }
  const providers: ProviderInfo[] = [];
  for (const raw of json.providers) {
    if (!isRecord(raw)) continue;
    const baseUrl = typeof raw.base_url === "string" ? raw.base_url : "";
    let path = "";
    const at = baseUrl.indexOf("/p/");
    if (at >= 0) path = baseUrl.slice(at);
    providers.push({
      id: typeof raw.id === "string" ? raw.id : "",
      name: typeof raw.name === "string" ? raw.name : "",
      wire: typeof raw.wire === "string" ? raw.wire : "",
      auth: typeof raw.auth === "string" ? raw.auth : "none",
      keyEnv: typeof raw.key_env === "string" ? raw.key_env : "",
      key: typeof raw.key === "string" ? raw.key : "",
      path,
      baseUrl,
    });
  }
  let manifest: WorkloadItem[] = [];
  if (isRecord(json.workload) && Array.isArray(json.workload.manifest)) {
    manifest = json.workload.manifest
      .filter(isRecord)
      .map((row) => ({
        id: typeof row.id === "string" ? row.id : "",
        api: typeof row.api === "string" ? row.api : "openai",
        stream: row.stream === true,
        weight: typeof row.weight === "number" ? row.weight : 1,
        requires: isRecord(row.requires) ? (row.requires as WorkloadItem["requires"]) : {},
      }));
  }
  return { providers, manifest };
}

function errorCode(wire: string, json: unknown): string | null {
  if (!isRecord(json)) return null;
  if (wire === "gemini") {
    if (isRecord(json.error) && typeof json.error.status === "string") return json.error.status;
    return null;
  }
  if (isRecord(json.error)) {
    if (typeof json.error.code === "string") return json.error.code;
    if (typeof json.error.type === "string") return json.error.type;
  }
  return null;
}

// One inference call against one provider with one model. The request
// is rebuilt in the provider's wire from the intermediate request; the
// answer is parsed back out of that wire. A non-2xx answer is a
// refusal outcome carrying the HTTP status the ledger judges by.
export async function callProvider(
  provider: ProviderInfo,
  ir: IrRequest,
  model: string,
): Promise<CallOutcome> {
  let url: string;
  let body: Record<string, unknown>;
  if (provider.wire === "openai") {
    url = `${provider.baseUrl}/v1/chat/completions`;
    body = buildOpenAiRequest(ir, model);
  } else if (provider.wire === "anthropic") {
    url = `${provider.baseUrl}/v1/messages`;
    body = buildAnthropicRequest(ir, model);
  } else if (provider.wire === "gemini") {
    url = `${provider.baseUrl}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    body = buildGeminiRequest(ir, model);
  } else {
    return { ok: false, status: 0, code: "unknown_wire" };
  }

  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...authHeaders(provider),
  };
  if (provider.wire === "anthropic") headers["anthropic-version"] = ANTHROPIC_VERSION;

  let response: Response;
  try {
    response = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
  } catch {
    return { ok: false, status: 0, code: "unreachable" };
  }

  const text = await response.text().catch(() => "");
  let json: unknown = null;
  try {
    json = JSON.parse(text);
  } catch {
    json = null;
  }

  if (response.status < 200 || response.status >= 300) {
    return { ok: false, status: response.status, code: errorCode(provider.wire, json) };
  }
  if (!isRecord(json)) return { ok: false, status: 0, code: "unparseable_answer" };

  const answer =
    provider.wire === "openai"
      ? parseOpenAiAnswer(json)
      : provider.wire === "anthropic"
        ? parseAnthropicAnswer(json)
        : parseGeminiAnswer(json);
  if (answer === null) return { ok: false, status: 0, code: "unparseable_answer" };
  if (answer.text === "" && answer.toolCalls.length === 0) {
    return { ok: false, status: 0, code: "empty_answer" };
  }
  return { ok: true, answer };
}
