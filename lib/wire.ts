// wire.ts — translation between the three provider wires (OpenAI Chat
// Completions, Anthropic Messages, Gemini generateContent) and the two
// client wires (OpenAI and Anthropic), through one intermediate shape.
//
// The intermediate request keeps exactly the fields the providers echo
// back for grading: the system text, the last user text, the tool names
// in order, max_tokens, the stop list and any image bytes. Every renderer
// below rebuilds those fields in the target wire so the echo a provider
// computes from our request matches the echo the client expects from
// its own request, field for field.

export type WireName = "openai" | "anthropic" | "gemini";
export type ClientWire = "openai" | "anthropic";

export interface TextPart {
  kind: "text";
  text: string;
}
export interface ImagePart {
  kind: "image";
  mediaType: string;
  data: string; // base64 payload, no data: prefix
}
export interface ImageUrlPart {
  kind: "image_url";
  url: string;
}
export interface ToolResultPart {
  kind: "tool_result";
  toolUseId: string;
  text: string;
  isError: boolean;
}
export type ContentPart = TextPart | ImagePart | ImageUrlPart | ToolResultPart;

export interface ToolCall {
  id: string;
  name: string;
  input: unknown; // parsed argument object when known
  argumentsJson: string; // the same arguments as one JSON string
}

export interface IrMessage {
  role: "user" | "assistant" | "tool";
  parts: ContentPart[];
  toolCalls: ToolCall[];
  toolCallId: string | null; // for role "tool": the call it answers
}

export interface IrTool {
  name: string;
  description: string;
  parameters: unknown;
}

export interface IrRequest {
  systemText: string;
  messages: IrMessage[];
  tools: IrTool[];
  toolChoice: unknown; // kept in the shape the client sent it
  toolChoiceWire: ClientWire | null; // which wire that shape belongs to
  maxTokens: number | null;
  temperature: number | null;
  stops: string[];
  jsonMode: boolean;
  responseFormat: unknown; // raw OpenAI response_format, when present
  stream: boolean;
}

export interface IrAnswer {
  text: string;
  stopReason: string | null;
  inputTokens: number | null;
  outputTokens: number | null;
  toolCalls: ToolCall[];
  id: string | null;
  created: number | null;
}

// ---------------------------------------------------------------- helpers

export function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  const out: string[] = [];
  for (const item of v) if (typeof item === "string") out.push(item);
  return out;
}

// Text of a string-or-blocks content value, joining text blocks with a
// newline — the same reading every wire's parser applies.
export function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("\n");
}

function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1], data: m[2] };
}

function argumentsToInput(argumentsJson: string): unknown {
  try {
    const v: unknown = JSON.parse(argumentsJson);
    return isRecord(v) ? v : { value: v };
  } catch {
    return {};
  }
}

// ------------------------------------------------------- client parsing

function openAiContentParts(content: unknown): ContentPart[] {
  if (typeof content === "string") return [{ kind: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ kind: "text", text: block.text });
    } else if (block.type === "image_url" && isRecord(block.image_url) && typeof block.image_url.url === "string") {
      const parsed = parseDataUrl(block.image_url.url);
      if (parsed !== null) parts.push({ kind: "image", mediaType: parsed.mediaType, data: parsed.data });
      else parts.push({ kind: "image_url", url: block.image_url.url });
    }
  }
  return parts;
}

export function parseOpenAiRequest(body: Record<string, unknown>): IrRequest {
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  const systemChunks: string[] = [];
  const messages: IrMessage[] = [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === "string" ? raw.role : "";
    if (role === "system") {
      const text = blocksText(raw.content);
      if (text !== "") systemChunks.push(text);
      continue;
    }
    if (role === "tool") {
      messages.push({
        role: "tool",
        parts: [{ kind: "text", text: blocksText(raw.content) }],
        toolCalls: [],
        toolCallId: typeof raw.tool_call_id === "string" ? raw.tool_call_id : null,
      });
      continue;
    }
    if (role !== "user" && role !== "assistant") continue;
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        if (typeof call.function.name !== "string") continue;
        const args = typeof call.function.arguments === "string" ? call.function.arguments : "{}";
        toolCalls.push({
          id: typeof call.id === "string" ? call.id : "",
          name: call.function.name,
          input: argumentsToInput(args),
          argumentsJson: args,
        });
      }
    }
    messages.push({ role, parts: openAiContentParts(raw.content), toolCalls, toolCallId: null });
  }
  const tools: IrTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      if (typeof tool.function.name !== "string") continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === "string" ? tool.function.description : "",
        parameters: tool.function.parameters ?? { type: "object", properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  const formatType = format !== null && typeof format.type === "string" ? format.type : null;
  return {
    systemText: systemChunks.join("\n"),
    messages,
    tools,
    toolChoice: body.tool_choice ?? null,
    toolChoiceWire: body.tool_choice !== undefined ? "openai" : null,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    temperature: asNumber(body.temperature),
    stops: stringList(body.stop),
    jsonMode: formatType === "json_object" || formatType === "json_schema",
    responseFormat: format,
    stream: body.stream === true,
  };
}

function anthropicContentParts(content: unknown, toolCalls: ToolCall[]): ContentPart[] {
  if (typeof content === "string") return [{ kind: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ kind: "text", text: block.text });
    } else if (block.type === "image" && isRecord(block.source)) {
      const source = block.source;
      if (source.type === "base64" && typeof source.data === "string") {
        parts.push({
          kind: "image",
          mediaType: typeof source.media_type === "string" ? source.media_type : "image/png",
          data: source.data,
        });
      } else if (source.type === "url" && typeof source.url === "string") {
        parts.push({ kind: "image_url", url: source.url });
      }
    } else if (block.type === "tool_use" && typeof block.name === "string") {
      const input = block.input ?? {};
      toolCalls.push({
        id: typeof block.id === "string" ? block.id : "",
        name: block.name,
        input,
        argumentsJson: JSON.stringify(input),
      });
    } else if (block.type === "tool_result") {
      parts.push({
        kind: "tool_result",
        toolUseId: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
        text: blocksText(block.content),
        isError: block.is_error === true,
      });
    }
  }
  return parts;
}

export function parseAnthropicRequest(body: Record<string, unknown>): IrRequest {
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  const messages: IrMessage[] = [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    if (raw.role !== "user" && raw.role !== "assistant") continue;
    const toolCalls: ToolCall[] = [];
    const parts = anthropicContentParts(raw.content, toolCalls);
    messages.push({ role: raw.role, parts, toolCalls, toolCallId: null });
  }
  const tools: IrTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== "string") continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === "string" ? tool.description : "",
        parameters: tool.input_schema ?? { type: "object", properties: {} },
      });
    }
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice: body.tool_choice ?? null,
    toolChoiceWire: body.tool_choice !== undefined ? "anthropic" : null,
    maxTokens: asNumber(body.max_tokens),
    temperature: asNumber(body.temperature),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    responseFormat: null,
    stream: body.stream === true,
  };
}

export function parseClientRequest(wire: ClientWire, body: Record<string, unknown>): IrRequest {
  return wire === "openai" ? parseOpenAiRequest(body) : parseAnthropicRequest(body);
}

// -------------------------------------------------- provider rendering

function openAiMessageContent(parts: ContentPart[]): unknown {
  const texts = parts.filter((p): p is TextPart => p.kind === "text");
  const others = parts.filter((p) => p.kind !== "text");
  if (others.length === 0) return texts.map((p) => p.text).join("\n");
  const blocks: unknown[] = [];
  for (const part of parts) {
    if (part.kind === "text") blocks.push({ type: "text", text: part.text });
    else if (part.kind === "image") {
      blocks.push({ type: "image_url", image_url: { url: `data:${part.mediaType};base64,${part.data}` } });
    } else if (part.kind === "image_url") {
      blocks.push({ type: "image_url", image_url: { url: part.url } });
    }
  }
  return blocks;
}

function openAiToolChoice(ir: IrRequest): unknown {
  if (ir.toolChoice === null) return undefined;
  if (ir.toolChoiceWire === "openai") return ir.toolChoice;
  if (isRecord(ir.toolChoice)) {
    const t = ir.toolChoice.type;
    if (t === "auto") return "auto";
    if (t === "any") return "required";
    if (t === "none") return "none";
    if (t === "tool" && typeof ir.toolChoice.name === "string") {
      return { type: "function", function: { name: ir.toolChoice.name } };
    }
  }
  return undefined;
}

export function buildOpenAiRequest(ir: IrRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (ir.systemText !== "") messages.push({ role: "system", content: ir.systemText });
  for (const msg of ir.messages) {
    if (msg.role === "tool") {
      messages.push({
        role: "tool",
        tool_call_id: msg.toolCallId ?? "",
        content: msg.parts.map((p) => (p.kind === "text" ? p.text : "")).join("\n"),
      });
      continue;
    }
    // Tool results riding inside a user message become tool messages of
    // their own; the remaining parts keep the user text untouched.
    const kept: ContentPart[] = [];
    for (const part of msg.parts) {
      if (part.kind === "tool_result" && msg.role === "user") {
        messages.push({ role: "tool", tool_call_id: part.toolUseId, content: part.text });
      } else {
        kept.push(part);
      }
    }
    if (msg.role === "user" && kept.length === 0 && msg.toolCalls.length === 0) continue;
    const out: Record<string, unknown> = { role: msg.role, content: openAiMessageContent(kept) };
    if (msg.toolCalls.length > 0) {
      out.tool_calls = msg.toolCalls.map((call) => ({
        id: call.id,
        type: "function",
        function: { name: call.name, arguments: call.argumentsJson },
      }));
      if (kept.every((p) => p.kind === "text" && p.text === "")) out.content = null;
    }
    messages.push(out);
  }
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (ir.maxTokens !== null) body.max_tokens = ir.maxTokens;
  if (ir.temperature !== null) body.temperature = ir.temperature;
  if (ir.stops.length > 0) body.stop = [...ir.stops];
  if (ir.tools.length > 0) {
    body.tools = ir.tools.map((tool) => ({
      type: "function",
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
    const choice = openAiToolChoice(ir);
    if (choice !== undefined) body.tool_choice = choice;
  }
  if (ir.jsonMode) body.response_format = ir.responseFormat ?? { type: "json_object" };
  return body;
}

function anthropicToolChoice(ir: IrRequest): unknown {
  if (ir.toolChoice === null) return undefined;
  if (ir.toolChoiceWire === "anthropic") return ir.toolChoice;
  if (typeof ir.toolChoice === "string") {
    if (ir.toolChoice === "auto") return { type: "auto" };
    if (ir.toolChoice === "required") return { type: "any" };
    if (ir.toolChoice === "none") return { type: "none" };
  }
  if (isRecord(ir.toolChoice) && isRecord(ir.toolChoice.function) && typeof ir.toolChoice.function.name === "string") {
    return { type: "tool", name: ir.toolChoice.function.name };
  }
  return undefined;
}

export function buildAnthropicRequest(ir: IrRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  for (const msg of ir.messages) {
    if (msg.role === "tool") {
      messages.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: msg.toolCallId ?? "",
            content: msg.parts.map((p) => (p.kind === "text" ? p.text : "")).join("\n"),
          },
        ],
      });
      continue;
    }
    const texts = msg.parts.filter((p): p is TextPart => p.kind === "text");
    const rich = msg.parts.some((p) => p.kind !== "text") || msg.toolCalls.length > 0;
    if (!rich) {
      messages.push({ role: msg.role, content: texts.map((p) => p.text).join("\n") });
      continue;
    }
    const blocks: Record<string, unknown>[] = [];
    for (const part of msg.parts) {
      if (part.kind === "text") {
        if (part.text !== "") blocks.push({ type: "text", text: part.text });
      } else if (part.kind === "image") {
        blocks.push({
          type: "image",
          source: { type: "base64", media_type: part.mediaType, data: part.data },
        });
      } else if (part.kind === "image_url") {
        blocks.push({ type: "image", source: { type: "url", url: part.url } });
      } else if (part.kind === "tool_result") {
        const block: Record<string, unknown> = {
          type: "tool_result",
          tool_use_id: part.toolUseId,
          content: part.text,
        };
        if (part.isError) block.is_error = true;
        blocks.push(block);
      }
    }
    for (const call of msg.toolCalls) {
      blocks.push({ type: "tool_use", id: call.id, name: call.name, input: call.input ?? {} });
    }
    messages.push({ role: msg.role, content: blocks });
  }
  const body: Record<string, unknown> = { model, messages };
  if (ir.systemText !== "") body.system = ir.systemText;
  if (ir.maxTokens !== null) body.max_tokens = ir.maxTokens;
  if (ir.temperature !== null) body.temperature = ir.temperature;
  if (ir.stops.length > 0) body.stop_sequences = [...ir.stops];
  if (ir.tools.length > 0) {
    body.tools = ir.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }));
    const choice = anthropicToolChoice(ir);
    if (choice !== undefined) body.tool_choice = choice;
  }
  return body;
}

export function buildGeminiRequest(ir: IrRequest, model: string): Record<string, unknown> {
  void model; // the model travels in the Gemini URL path, not the body
  const contents: Record<string, unknown>[] = [];
  for (const msg of ir.messages) {
    const parts: Record<string, unknown>[] = [];
    if (msg.role === "tool") {
      parts.push({
        functionResponse: {
          name: msg.toolCallId ?? "",
          response: { result: msg.parts.map((p) => (p.kind === "text" ? p.text : "")).join("\n") },
        },
      });
      contents.push({ role: "user", parts });
      continue;
    }
    for (const part of msg.parts) {
      if (part.kind === "text") {
        if (part.text !== "") parts.push({ text: part.text });
      } else if (part.kind === "image") {
        parts.push({ inlineData: { mimeType: part.mediaType, data: part.data } });
      } else if (part.kind === "image_url") {
        parts.push({ fileData: { fileUri: part.url } });
      } else if (part.kind === "tool_result") {
        parts.push({
          functionResponse: { name: part.toolUseId, response: { result: part.text } },
        });
      }
    }
    for (const call of msg.toolCalls) {
      parts.push({ functionCall: { name: call.name, args: call.input ?? {} } });
    }
    if (parts.length === 0) parts.push({ text: "" });
    contents.push({ role: msg.role === "assistant" ? "model" : "user", parts });
  }
  const body: Record<string, unknown> = { contents };
  if (ir.systemText !== "") body.systemInstruction = { parts: [{ text: ir.systemText }] };
  if (ir.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: ir.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
  }
  const config: Record<string, unknown> = {};
  if (ir.maxTokens !== null) config.maxOutputTokens = ir.maxTokens;
  if (ir.temperature !== null) config.temperature = ir.temperature;
  if (ir.stops.length > 0) config.stopSequences = [...ir.stops];
  if (ir.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// --------------------------------------------------- provider answers

export function parseOpenAiAnswer(json: Record<string, unknown>): IrAnswer | null {
  const choices = Array.isArray(json.choices) ? json.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (first === null || !isRecord(first.message)) return null;
  const message = first.message;
  let text = "";
  if (typeof message.content === "string") text = message.content;
  else text = blocksText(message.content);
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      if (typeof call.function.name !== "string") continue;
      const args = typeof call.function.arguments === "string" ? call.function.arguments : "{}";
      toolCalls.push({
        id: typeof call.id === "string" ? call.id : "",
        name: call.function.name,
        input: argumentsToInput(args),
        argumentsJson: args,
      });
    }
  }
  const usage = isRecord(json.usage) ? json.usage : null;
  return {
    text,
    stopReason: typeof first.finish_reason === "string" ? first.finish_reason : null,
    inputTokens: usage !== null ? asNumber(usage.prompt_tokens) : null,
    outputTokens: usage !== null ? asNumber(usage.completion_tokens) : null,
    toolCalls,
    id: typeof json.id === "string" ? json.id : null,
    created: asNumber(json.created),
  };
}

export function parseAnthropicAnswer(json: Record<string, unknown>): IrAnswer | null {
  if (!Array.isArray(json.content)) return null;
  let text = "";
  const toolCalls: ToolCall[] = [];
  for (const block of json.content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") text += block.text;
    else if (block.type === "tool_use" && typeof block.name === "string") {
      const input = block.input ?? {};
      toolCalls.push({
        id: typeof block.id === "string" ? block.id : "",
        name: block.name,
        input,
        argumentsJson: JSON.stringify(input),
      });
    }
  }
  const usage = isRecord(json.usage) ? json.usage : null;
  return {
    text,
    stopReason: typeof json.stop_reason === "string" ? json.stop_reason : null,
    inputTokens: usage !== null ? asNumber(usage.input_tokens) : null,
    outputTokens: usage !== null ? asNumber(usage.output_tokens) : null,
    toolCalls,
    id: typeof json.id === "string" ? json.id : null,
    created: null,
  };
}

export function parseGeminiAnswer(json: Record<string, unknown>): IrAnswer | null {
  const candidates = Array.isArray(json.candidates) ? json.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null || !isRecord(first.content)) return null;
  const parts = Array.isArray(first.content.parts) ? first.content.parts : [];
  let text = "";
  const toolCalls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === "string") text += part.text;
    else if (isRecord(part.functionCall) && typeof part.functionCall.name === "string") {
      const args = part.functionCall.args ?? {};
      toolCalls.push({
        id: "",
        name: part.functionCall.name,
        input: args,
        argumentsJson: JSON.stringify(args),
      });
    }
  }
  const usage = isRecord(json.usageMetadata) ? json.usageMetadata : null;
  return {
    text,
    stopReason: typeof first.finishReason === "string" ? first.finishReason : null,
    inputTokens: usage !== null ? asNumber(usage.promptTokenCount) : null,
    outputTokens: usage !== null ? asNumber(usage.candidatesTokenCount) : null,
    toolCalls,
    id: null,
    created: null,
  };
}

// ------------------------------------------------------ client answers

const FALLBACK_CREATED = 1767225600; // a fixed epoch: no wall clock in answers

export function toOpenAiStop(reason: string | null): string {
  switch (reason) {
    case "stop":
    case "end_turn":
    case "STOP":
      return "stop";
    case "length":
    case "max_tokens":
    case "MAX_TOKENS":
      return "length";
    case "tool_calls":
    case "function_call":
    case "tool_use":
      return "tool_calls";
    case "content_filter":
    case "refusal":
    case "SAFETY":
      return "content_filter";
    default:
      return "stop";
  }
}

export function toAnthropicStop(reason: string | null): string {
  switch (reason) {
    case "end_turn":
    case "stop":
    case "STOP":
      return "end_turn";
    case "max_tokens":
    case "MAX_TOKENS":
    case "length":
      return "max_tokens";
    case "tool_use":
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "refusal":
    case "SAFETY":
    case "content_filter":
      return "refusal";
    default:
      return "end_turn";
  }
}

function openAiUsage(answer: IrAnswer): Record<string, number> {
  const input = answer.inputTokens ?? 0;
  const output = answer.outputTokens ?? 0;
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

export function renderOpenAiJson(answer: IrAnswer, model: string): Record<string, unknown> {
  const message: Record<string, unknown> = { role: "assistant", content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id,
      type: "function",
      function: { name: call.name, arguments: call.argumentsJson },
    }));
  }
  const finish = answer.toolCalls.length > 0 ? "tool_calls" : toOpenAiStop(answer.stopReason);
  return {
    id: answer.id ?? "chatcmpl-sim",
    object: "chat.completion",
    created: answer.created ?? FALLBACK_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: openAiUsage(answer),
  };
}

export function renderAnthropicJson(answer: IrAnswer, model: string): Record<string, unknown> {
  const content: Record<string, unknown>[] = [];
  if (answer.text !== "") content.push({ type: "text", text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: "tool_use", id: call.id, name: call.name, input: call.input ?? {} });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const stop = answer.toolCalls.length > 0 ? "tool_use" : toAnthropicStop(answer.stopReason);
  return {
    id: answer.id ?? "msg_sim",
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: answer.outputTokens ?? 0 },
  };
}

// Split an answer text into deterministic stream pieces. The piece
// size is fixed so a replayed stream is byte for byte the live one.
const STREAM_PIECE_CHARS = 13;

export function streamPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) {
    pieces.push(text.slice(at, at + STREAM_PIECE_CHARS));
  }
  return pieces.length === 0 ? [""] : pieces;
}

function sseData(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function sseEvent(event: string, payload: unknown): string {
  return `event: ${event}\n${sseData(payload)}`;
}

export function renderOpenAiSse(answer: IrAnswer, model: string): string {
  const id = answer.id ?? "chatcmpl-sim";
  const created = answer.created ?? FALLBACK_CREATED;
  const chunk = (delta: Record<string, unknown>, finish: string | null): Record<string, unknown> => ({
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseData(chunk({ role: "assistant", content: "" }, null));
  if (answer.toolCalls.length > 0) {
    const calls = answer.toolCalls.map((call, i) => ({
      index: i,
      id: call.id,
      type: "function",
      function: { name: call.name, arguments: call.argumentsJson },
    }));
    out += sseData(chunk({ tool_calls: calls }, null));
    out += sseData(chunk({}, "tool_calls"));
  } else {
    for (const piece of streamPieces(answer.text)) {
      out += sseData(chunk({ content: piece }, null));
    }
    out += sseData(chunk({}, toOpenAiStop(answer.stopReason)));
  }
  out += sseData({
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [],
    usage: openAiUsage(answer),
  });
  out += "data: [DONE]\n\n";
  return out;
}

export function renderAnthropicSse(answer: IrAnswer, model: string): string {
  const id = answer.id ?? "msg_sim";
  let out = sseEvent("message_start", {
    type: "message_start",
    message: {
      id,
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== "") {
    out += sseEvent("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "text", text: "" },
    });
    for (const piece of streamPieces(answer.text)) {
      out += sseEvent("content_block_delta", {
        type: "content_block_delta",
        index,
        delta: { type: "text_delta", text: piece },
      });
    }
    out += sseEvent("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    out += sseEvent("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id: call.id, name: call.name, input: {} },
    });
    out += sseEvent("content_block_delta", {
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json: call.argumentsJson },
    });
    out += sseEvent("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  if (index === 0) {
    out += sseEvent("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    });
    out += sseEvent("content_block_stop", { type: "content_block_stop", index: 0 });
  }
  const stop = answer.toolCalls.length > 0 ? "tool_use" : toAnthropicStop(answer.stopReason);
  out += sseEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: answer.outputTokens ?? 0 },
  });
  out += sseEvent("message_stop", { type: "message_stop" });
  return out;
}
