// main.ts — a free-model aggregator: one HTTP server speaking the
// OpenAI Chat Completions and Anthropic Messages APIs, routing every
// request to the best free model that can serve it across the trial's
// ten providers, with rating-ranked failover when a provider refuses.
//
//   node main.ts --list   print the catalog document and exit 0
//   node main.ts          serve one trial: read the brief, probe the
//                         providers, register with the env, answer the
//                         client's requests, exit 0 when /work is done
//
// Every choice derives from the brief, the requests and the providers'
// answers — never the wall clock and never unseeded randomness — so a
// replayed trial leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";

import {
  isRecord,
  parseClientRequest,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAiJson,
  renderOpenAiSse,
  type ClientWire,
  type IrAnswer,
} from "./lib/wire.ts";
import {
  callProvider,
  fetchInstance,
  probeProvider,
  type CatalogModel,
  type ProbeResult,
  type ProviderInfo,
} from "./lib/providers.ts";
import {
  buildPlan,
  feasibleCandidates,
  rankCandidates,
  servingOrder,
  Ledger,
  type Plan,
  type RatingRow,
  type Requirements,
  type WorkloadItem,
} from "./lib/plan.ts";

const MAX_ATTEMPTS = 8;
const FIXED_CREATED = 1767225600;

interface Brief {
  directory: ProviderInfo[];
  ratings: RatingRow[];
  manifest: WorkloadItem[];
}

interface CatalogDoc {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }[];
}

// ------------------------------------------------------------ the brief

function extractBriefDocs(text: string): Map<string, Record<string, unknown>> {
  const docs = new Map<string, Record<string, unknown>>();
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    try {
      const value: unknown = JSON.parse(trimmed);
      if (isRecord(value) && typeof value.schema === "string") docs.set(value.schema, value);
    } catch {
      // not one of the brief's JSON documents; prose lines land here
    }
  }
  return docs;
}

function briefProviders(doc: Record<string, unknown> | undefined, envUrl: string): ProviderInfo[] {
  if (doc === undefined || !Array.isArray(doc.providers)) return [];
  const out: ProviderInfo[] = [];
  for (const raw of doc.providers) {
    if (!isRecord(raw)) continue;
    const path = typeof raw.path === "string" ? raw.path : "";
    out.push({
      id: typeof raw.id === "string" ? raw.id : "",
      name: typeof raw.name === "string" ? raw.name : "",
      wire: typeof raw.wire === "string" ? raw.wire : "",
      auth: typeof raw.auth === "string" ? raw.auth : "none",
      keyEnv: typeof raw.key_env === "string" ? raw.key_env : "",
      key: typeof raw.key === "string" ? raw.key : "",
      path,
      baseUrl: envUrl + path,
    });
  }
  return out;
}

function briefRatings(doc: Record<string, unknown> | undefined): RatingRow[] {
  if (doc === undefined || !Array.isArray(doc.ratings)) return [];
  const out: RatingRow[] = [];
  for (const raw of doc.ratings) {
    if (!isRecord(raw)) continue;
    out.push({
      model: typeof raw.model === "string" ? raw.model : "",
      provider: typeof raw.provider === "string" ? raw.provider : "",
      rating_milli: typeof raw.rating_milli === "number" ? raw.rating_milli : 0,
      context_length: typeof raw.context_length === "number" ? raw.context_length : null,
      tools: raw.tools === true,
      vision: raw.vision === true,
      json: raw.json === true,
      free: raw.free === true,
    });
  }
  return out;
}

function manifestOf(doc: Record<string, unknown> | undefined): WorkloadItem[] {
  if (doc === undefined || !Array.isArray(doc.manifest)) return [];
  const out: WorkloadItem[] = [];
  for (const raw of doc.manifest) {
    if (!isRecord(raw)) continue;
    out.push({
      id: typeof raw.id === "string" ? raw.id : "",
      api: typeof raw.api === "string" ? raw.api : "openai",
      stream: raw.stream === true,
      weight: typeof raw.weight === "number" ? raw.weight : 1,
      requires: isRecord(raw.requires) ? (raw.requires as Requirements) : {},
    });
  }
  return out;
}

async function readBrief(envUrl: string): Promise<Brief | null> {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file === undefined || file === "") return null;
  let text: string;
  try {
    text = await readFile(file, "utf8");
  } catch {
    return null;
  }
  const docs = extractBriefDocs(text);
  const directory = briefProviders(docs.get("free-model-aggregator-01/providers/v1"), envUrl);
  if (directory.length === 0) return null;
  return {
    directory,
    ratings: briefRatings(docs.get("free-model-aggregator-01/ratings/v1")),
    manifest: manifestOf(docs.get("free-model-aggregator-01/workload/v1")),
  };
}

// ------------------------------------------------------------- catalog

async function probeAll(providers: ProviderInfo[]): Promise<Map<string, ProbeResult>> {
  const probes = new Map<string, ProbeResult>();
  for (const provider of providers) {
    probes.set(provider.id, await probeProvider(provider));
  }
  return probes;
}

function buildCatalog(
  directory: ProviderInfo[],
  probes: Map<string, ProbeResult>,
  ratings: RatingRow[],
): CatalogDoc {
  const providers = directory.map((provider) => {
    const probe = probes.get(provider.id);
    let models: CatalogModel[];
    if (probe !== undefined && probe.ok) {
      models = probe.models;
    } else {
      // The probe failed: fall back to the brief's quality table, which
      // lists this provider's free models with their context lengths.
      models = ratings
        .filter((row) => row.provider === provider.id && row.free)
        .map((row) => ({ id: row.model, free: true, context_length: row.context_length }));
    }
    return {
      id: provider.id,
      base_url: provider.path,
      key_env: provider.keyEnv,
      wire: provider.wire,
      models,
    };
  });
  return { providers };
}

function freeModelIds(catalog: CatalogDoc): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !seen.has(model.id)) {
        seen.add(model.id);
        ids.push(model.id);
      }
    }
  }
  return ids;
}

// -------------------------------------------------------------- serving

interface TrialState {
  providers: Map<string, ProviderInfo>;
  wires: Map<string, string>;
  ratings: RatingRow[];
  manifest: WorkloadItem[];
  probes: Map<string, ProbeResult>;
  probedFree: Set<string>; // "provider/model" pairs the probe listed as free
  plan: Plan | null;
  ledger: Ledger;
  servedCount: number;
}

function requirementsFromBody(body: Record<string, unknown>, wire: ClientWire): Requirements {
  const requires: Requirements = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (isRecord(body.response_format)) requires.json = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (wire === "openai" && block.type === "image_url") requires.vision = true;
      if (wire === "anthropic" && block.type === "image") requires.vision = true;
    }
  }
  return requires;
}

async function serveOne(
  state: TrialState,
  wire: ClientWire,
  body: Record<string, unknown>,
): Promise<{ status: number; contentType: string; payload: string }> {
  const item = state.manifest[state.servedCount] ?? null;
  state.servedCount += 1;
  const ir = parseClientRequest(wire, body);
  const requires: Requirements = item !== null ? item.requires : requirementsFromBody(body, wire);
  const jsonMode = requires.json === true || ir.jsonMode;

  let candidates = feasibleCandidates(state.ratings, state.wires, requires, jsonMode);
  candidates = candidates.filter((cand) => {
    const probe = state.probes.get(cand.provider);
    if (probe === undefined || !probe.ok) return false;
    return state.probedFree.has(`${cand.provider}/${cand.model}`);
  });
  const ranked = rankCandidates(candidates, state.ledger, wire);
  const order = servingOrder(ranked, state.plan, item !== null ? item.id : null);

  let answer: IrAnswer | null = null;
  let servedModel = "";
  let attempts = 0;
  for (const cand of order) {
    if (attempts >= MAX_ATTEMPTS) break;
    if (state.ledger.isRetired(cand.provider)) continue;
    const provider = state.providers.get(cand.provider);
    if (provider === undefined) continue;
    attempts += 1;
    state.ledger.recordAttempt(cand.provider);
    const outcome = await callProvider(provider, ir, cand.model);
    if (outcome.ok) {
      answer = outcome.answer;
      servedModel = cand.model;
      break;
    }
    state.ledger.recordRefusal(cand.provider, outcome.status);
  }

  if (answer === null) {
    const message = "no provider could serve this request";
    if (wire === "openai") {
      return {
        status: 502,
        contentType: "application/json",
        payload: JSON.stringify({
          error: { message, type: "api_error", param: null, code: "all_providers_failed" },
        }),
      };
    }
    return {
      status: 502,
      contentType: "application/json",
      payload: JSON.stringify({ type: "error", error: { type: "api_error", message } }),
    };
  }

  if (wire === "openai") {
    if (ir.stream) {
      return {
        status: 200,
        contentType: "text/event-stream; charset=utf-8",
        payload: renderOpenAiSse(answer, servedModel),
      };
    }
    return { status: 200, contentType: "application/json", payload: JSON.stringify(renderOpenAiJson(answer, servedModel)) };
  }
  if (ir.stream) {
    return {
      status: 200,
      contentType: "text/event-stream; charset=utf-8",
      payload: renderAnthropicSse(answer, servedModel),
    };
  }
  return {
    status: 200,
    contentType: "application/json",
    payload: JSON.stringify(renderAnthropicJson(answer, servedModel)),
  };
}

function modelsPayload(wire: ClientWire, ids: string[]): string {
  if (wire === "anthropic") {
    return JSON.stringify({
      data: ids.map((id) => ({
        type: "model",
        id,
        display_name: id,
        created_at: "2026-01-01T00:00:00Z",
      })),
      has_more: false,
    });
  }
  return JSON.stringify({
    object: "list",
    data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "aggregator" })),
  });
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > 4 * 1024 * 1024) throw new Error("request body too large");
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function sendJson(res: ServerResponse, status: number, payload: string): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(payload);
}

// ---------------------------------------------------------------- trial

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function runTrial(): Promise<void> {
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  const brief = await readBrief(envUrl);
  if (brief === null) throw new Error("no readable brief at MAKEFUN_PROMPT_FILE");

  // The instance directory carries absolute base URLs for dialling and
  // the same manifest the brief carries; retry briefly in case the env
  // is still coming up.
  let instance: Awaited<ReturnType<typeof fetchInstance>> | null = null;
  for (let attempt = 0; attempt < 20 && instance === null; attempt++) {
    try {
      instance = await fetchInstance(envUrl);
    } catch {
      await sleep(300);
    }
  }
  if (instance === null) throw new Error("the env's /instance route never answered");

  const directory = instance.providers.length > 0 ? instance.providers : brief.directory;
  const byId = new Map(directory.map((p) => [p.id, p]));
  // Fill any path the instance form could not derive from the brief's.
  for (const provider of directory) {
    if (provider.path === "") {
      const fromBrief = brief.directory.find((p) => p.id === provider.id);
      if (fromBrief !== undefined) provider.path = fromBrief.path;
    }
  }
  const manifest = instance.manifest.length > 0 ? instance.manifest : brief.manifest;

  const probes = await probeAll(directory);
  const catalog = buildCatalog(directory, probes, brief.ratings);

  const wires = new Map(directory.map((p) => [p.id, p.wire]));
  const probeOk = new Map([...probes.entries()].map(([id, probe]) => [id, probe.ok]));
  const plan = buildPlan(brief.ratings, manifest, wires, probeOk);
  if (plan !== null) {
    for (const item of manifest) {
      const entry = plan.get(item.id);
      if (entry !== undefined) {
        process.stderr.write(`plan ${item.id} ${entry.provider} ${entry.model} ${entry.rating_milli}\n`);
      }
    }
  } else {
    process.stderr.write("plan none (serving by ranking alone)\n");
  }

  const probedFree = new Set<string>();
  for (const [id, probe] of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      if (model.free) probedFree.add(`${id}/${model.id}`);
    }
  }

  const state: TrialState = {
    providers: byId,
    wires,
    ratings: brief.ratings,
    manifest,
    probes,
    probedFree,
    plan,
    ledger: new Ledger(),
    servedCount: 0,
  };
  const modelIds = freeModelIds(catalog);

  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://entry");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const wire: ClientWire = req.headers["x-api-key"] !== undefined ? "anthropic" : "openai";
          sendJson(res, 200, modelsPayload(wire, modelIds));
          return;
        }
        if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
          const wire: ClientWire = url.pathname === "/v1/chat/completions" ? "openai" : "anthropic";
          const raw = await readBody(req);
          let body: Record<string, unknown>;
          try {
            const parsed: unknown = JSON.parse(raw);
            if (!isRecord(parsed)) throw new Error("not an object");
            body = parsed;
          } catch {
            const message = "the request body is not one JSON object";
            sendJson(
              res,
              400,
              wire === "openai"
                ? JSON.stringify({ error: { message, type: "invalid_request_error", param: null, code: "invalid_json" } })
                : JSON.stringify({ type: "error", error: { type: "invalid_request_error", message } }),
            );
            return;
          }
          const served = await serveOne(state, wire, body);
          res.writeHead(served.status, { "content-type": served.contentType });
          res.end(served.payload);
          return;
        }
        sendJson(res, 404, JSON.stringify({ error: { message: "no such route", type: "invalid_request_error", param: null, code: "not_found" } }));
      } catch (err) {
        const message = err instanceof Error ? err.message : "internal error";
        if (!res.headersSent) {
          sendJson(res, 500, JSON.stringify({ error: { message, type: "api_error", param: null, code: "internal" } }));
        } else {
          res.end();
        }
      }
    })();
  });

  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("the server did not bind a port");
  const port = address.port;

  const registerResponse = await fetch(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
  });
  if (registerResponse.status < 200 || registerResponse.status >= 300) {
    const detail = await registerResponse.text().catch(() => "");
    throw new Error(`POST /register answered ${registerResponse.status}: ${detail}`);
  }
  await registerResponse.text().catch(() => "");

  // Poll /work until the client has sent every request, then exit 0.
  for (let polls = 0; polls < 3000; polls++) {
    let done = false;
    try {
      const workResponse = await fetch(`${envUrl}/work`);
      if (workResponse.status >= 200 && workResponse.status < 300) {
        const work: unknown = await workResponse.json().catch(() => null);
        if (isRecord(work) && work.done === true) done = true;
      }
    } catch {
      // a transient env error is not the end of the workload
    }
    if (done) break;
    await sleep(300);
  }

  await new Promise<void>((resolve) => server.close(() => resolve()));
}

// ----------------------------------------------------------------- main

async function main(): Promise<void> {
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  if (process.argv.includes("--list")) {
    if (envUrl === "") {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      return;
    }
    const brief = await readBrief(envUrl);
    if (brief === null) {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      return;
    }
    const probes = await probeAll(brief.directory);
    const catalog = buildCatalog(brief.directory, probes, brief.ratings);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  await runTrial();
}

main().then(
  () => {
    process.exit(0);
  },
  (err: unknown) => {
    process.stderr.write(`free-model-aggregator: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  },
);
