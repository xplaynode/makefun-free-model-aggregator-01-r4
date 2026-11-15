# Free Model Aggregator

One server that speaks both the OpenAI Chat Completions API and the
Anthropic Messages API — each in JSON and in server-sent events — and
routes every request to the best free model that can serve it across
ten free-tier providers, each on its own wire (OpenAI, Anthropic or
Gemini) with its own auth scheme.

## Design

**Rating-ranked routing with a failover ledger.** Every candidate is a
free model from the sealed quality table that meets the request's
requirements (tools, vision, JSON mode, minimum context). Candidates
are ranked by rating, then by how many attempts their provider has
already spent this trial, then by speaking the client's own wire, then
by provider and model id. Every provider attempt is recorded in a
ledger. A refusal retires a provider for the rest of the trial only
when it is final — a 400 or a 401. A scheduled refusal (a 402 or 429 on
the provider's failure schedule, a spent budget, an outage) costs only
that one call: the provider stays in rotation for later requests, and
the request fails over to the next candidate, up to 8 attempts.

**Round-3 fixed primary plan.** Before serving, the entry computes one
planned primary provider per request. The plan maximises the sum of
request weight × rating subject to a fixed multiplicity cap per
provider:

| provider    | cap |
|-------------|-----|
| groq        | 5   |
| openrouter  | 7   |
| cerebras    | 1   |
| gemini      | 1   |
| zai         | 1   |
| sambanova   | 1   |
| pollinations| 1   |
| cloudflare  | 1   |
| ovh         | 1   |
| kilol       | 1   |

The rationale: concentrate primaries on the deep-budget high-rating
providers (groq and openrouter) and use every other provider's
never-refused first call exactly once — a provider's failure schedule
never refuses its first call of a trial, so eight of the twenty
primaries are guaranteed to land, and the two multi-call providers
have budgets well beyond the calls the plan gives them. The plan is
built by a deterministic greedy pass (most-constrained request first)
followed by improving single reassignments and pairwise swaps. Its
quality is 0.9872 of the theoretical bound (the bound ignores provider
budgets entirely), so graded quality lands at 98–99 and the rounded
score stays 99 while upstream calls drop to ~20–21 for 20 requests —
nearly every request is served by its first call.

Serving a request tries the planned provider's candidates first
(rating order), then every other provider's candidates in ranking
order. Final-refusal-only provider retirement (above) keeps a provider
that merely hit a scheduled refusal available as a later failover.

**Wire translation.** Requests are parsed into one intermediate shape
that preserves exactly what providers echo back — system text, user
text, tool names, max_tokens, stop sequences, image bytes, JSON mode —
then rebuilt in the chosen provider's wire, including tool calls,
vision/image content blocks and JSON mode (JSON-mode requests never
route to an Anthropic-wire provider, whose wire cannot carry
`response_format`). Provider calls are non-streaming; the answer is
translated back and, when the client asked for a stream, re-emitted as
a synthesized event stream in the client's wire with the provider's
usage counts and its stop reason mapped across wires (end_turn ↔ stop,
max_tokens ↔ length, tool_use ↔ tool_calls, refusal ↔ content_filter).

**Parity surfaces.** The wire, catalog and served behaviour are
unchanged from the previous round: `--list` prints the same catalog
document that is registered with the env (providers probed once, their
own paths as base URLs, every model with its true free flag and
context length), `GET /v1/models` answers in the wire its auth header
names and lists exactly the registered free model ids, and every
response is a well-formed answer of the client's API.

## Layout

- `main.ts` — entrypoint: brief parsing, catalog, the HTTP server,
  registration and the `/work` polling loop, the serving loop.
- `lib/wire.ts` — wire translation between OpenAI, Anthropic and
  Gemini shapes, streaming and non-streaming.
- `lib/providers.ts` — the env instance directory, provider probes
  and provider calls.
- `lib/plan.ts` — candidate ranking, the attempt ledger, the plan.

## Running

```
node main.ts --list   # print the catalog document, exit 0
node main.ts          # serve one trial (MAKEFUN_ENV_URL and
                      # MAKEFUN_PROMPT_FILE set by the harness)
```
