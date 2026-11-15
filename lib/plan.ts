// plan.ts — candidate ranking, the attempt ledger, and the fixed
// primary plan that assigns one planned provider to every request.
//
// Ranking is by sealed rating first. The ledger counts every provider
// attempt of the trial and remembers providers retired by a final
// refusal; it is the tie-breaker that spreads repeat picks across
// providers instead of hammering the one that already answered.
//
// The plan spends a fixed multiplicity budget per provider (PLAN_CAPS):
// the two deep-budget, high-rating providers take most primaries, and
// every other provider takes exactly one — its first call of a trial,
// which a provider's failure schedule never refuses. A deterministic
// greedy pass followed by improving reassignments and pairwise swaps
// maximises the weight-times-rating sum under those caps.

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadItem {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: Requirements;
}

export interface Candidate {
  provider: string;
  model: string;
  rating_milli: number;
  wire: string;
}

export interface PlanEntry {
  provider: string;
  model: string;
  rating_milli: number;
}

export type Plan = Map<string, PlanEntry>;

export const PLAN_CAPS: Record<string, number> = {
  groq: 5,
  cerebras: 1,
  gemini: 1,
  zai: 1,
  openrouter: 7,
  sambanova: 1,
  pollinations: 1,
  cloudflare: 1,
  ovh: 1,
  kilol: 1,
};

// ------------------------------------------------------------- ledger

export class Ledger {
  private attempts = new Map<string, number>();
  private retired = new Set<string>();

  recordAttempt(provider: string): void {
    this.attempts.set(provider, (this.attempts.get(provider) ?? 0) + 1);
  }

  // A refusal retires a provider for the rest of the trial only when it
  // is final: a 400 (the call itself was rejected — an unknown or paid
  // model on our side) or a 401 (the key was refused). A provider whose
  // budget is spent or which is in outage refuses every later call the
  // same way, so retiring it early would change nothing; a scheduled
  // refusal (402 / 429 on the provider's failure schedule) costs only
  // that one call and the provider stays in rotation.
  recordRefusal(provider: string, status: number): boolean {
    if (status === 400 || status === 401) {
      this.retired.add(provider);
      return true;
    }
    return false;
  }

  count(provider: string): number {
    return this.attempts.get(provider) ?? 0;
  }

  isRetired(provider: string): boolean {
    return this.retired.has(provider);
  }
}

// ---------------------------------------------------------- feasibility

export function modelMeets(row: RatingRow, requires: Requirements): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined) {
    if (row.context_length === null || row.context_length < requires.min_context) return false;
  }
  return true;
}

// Feasible candidates for one request: free models that meet the
// manifest requirements. A JSON-mode request additionally excludes
// models on an Anthropic-wire provider, because that wire cannot carry
// the response_format the request's echo is checked against.
export function feasibleCandidates(
  ratings: RatingRow[],
  wires: Map<string, string>,
  requires: Requirements,
  jsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const row of ratings) {
    if (!row.free) continue;
    if (!modelMeets(row, requires)) continue;
    const wire = wires.get(row.provider) ?? "";
    if (jsonMode && wire === "anthropic") continue;
    out.push({ provider: row.provider, model: row.model, rating_milli: row.rating_milli, wire });
  }
  return out;
}

function compareCandidates(
  a: Candidate,
  b: Candidate,
  ledger: Ledger,
  clientWire: string,
): number {
  if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
  const byAttempts = ledger.count(a.provider) - ledger.count(b.provider);
  if (byAttempts !== 0) return byAttempts;
  const aSame = a.wire === clientWire ? 0 : 1;
  const bSame = b.wire === clientWire ? 0 : 1;
  if (aSame !== bSame) return aSame - bSame;
  if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
  if (a.model !== b.model) return a.model < b.model ? -1 : 1;
  return 0;
}

export function rankCandidates(
  candidates: Candidate[],
  ledger: Ledger,
  clientWire: string,
): Candidate[] {
  const live = candidates.filter((c) => !ledger.isRetired(c.provider));
  return live.sort((a, b) => compareCandidates(a, b, ledger, clientWire));
}

// -------------------------------------------------------------- plan

interface ProviderBest {
  provider: string;
  model: string;
  rating_milli: number;
}

// The best feasible model of each provider for one request, providers
// in id order so every later choice is deterministic.
function bestPerProvider(
  ratings: RatingRow[],
  wires: Map<string, string>,
  item: WorkloadItem,
): ProviderBest[] {
  const feasible = feasibleCandidates(ratings, wires, item.requires, item.requires.json === true);
  const byProvider = new Map<string, ProviderBest>();
  for (const cand of feasible) {
    const cur = byProvider.get(cand.provider);
    if (
      cur === undefined ||
      cand.rating_milli > cur.rating_milli ||
      (cand.rating_milli === cur.rating_milli && cand.model < cur.model)
    ) {
      byProvider.set(cand.provider, {
        provider: cand.provider,
        model: cand.model,
        rating_milli: cand.rating_milli,
      });
    }
  }
  return [...byProvider.values()].sort((a, b) => (a.provider < b.provider ? -1 : 1));
}

export function buildPlan(
  ratings: RatingRow[] | null,
  workload: WorkloadItem[] | null,
  wires: Map<string, string>,
  probes: Map<string, boolean>,
): Plan | null {
  if (ratings === null || workload === null || workload.length === 0) return null;

  const caps = new Map<string, number>();
  for (const [provider, cap] of Object.entries(PLAN_CAPS)) {
    caps.set(provider, probes.get(provider) === false ? 0 : cap);
  }
  let totalCap = 0;
  for (const cap of caps.values()) totalCap += cap;
  if (totalCap < workload.length) return null;

  const options = new Map<string, ProviderBest[]>();
  for (const item of workload) {
    const best = bestPerProvider(ratings, wires, item);
    if (best.length === 0) return null;
    options.set(item.id, best);
  }
  const bestOf = (requestId: string, provider: string): ProviderBest | null => {
    const list = options.get(requestId) ?? [];
    for (const entry of list) if (entry.provider === provider) return entry;
    return null;
  };

  // Greedy: most-constrained request first — the one with the fewest
  // providers whose rating sits within 40 milli of its best rating.
  const order = [...workload].sort((a, b) => {
    const listA = options.get(a.id) ?? [];
    const listB = options.get(b.id) ?? [];
    const topA = Math.max(...listA.map((e) => e.rating_milli));
    const topB = Math.max(...listB.map((e) => e.rating_milli));
    const nearA = listA.filter((e) => e.rating_milli >= topA - 40).length;
    const nearB = listB.filter((e) => e.rating_milli >= topB - 40).length;
    if (nearA !== nearB) return nearA - nearB;
    return a.id < b.id ? -1 : 1;
  });

  const assignment = new Map<string, PlanEntry>();
  const used = new Map<string, number>();
  const usedCount = (provider: string): number => used.get(provider) ?? 0;

  for (const item of order) {
    const choices = (options.get(item.id) ?? [])
      .filter((e) => usedCount(e.provider) < (caps.get(e.provider) ?? 0))
      .sort((x, y) => {
        if (x.rating_milli !== y.rating_milli) return y.rating_milli - x.rating_milli;
        return x.provider < y.provider ? -1 : 1;
      });
    if (choices.length === 0) return null;
    const pick = choices[0];
    assignment.set(item.id, { provider: pick.provider, model: pick.model, rating_milli: pick.rating_milli });
    used.set(pick.provider, usedCount(pick.provider) + 1);
  }

  const weightOf = new Map(workload.map((item) => [item.id, item.weight]));
  const valueOf = (requestId: string, entry: PlanEntry): number =>
    (weightOf.get(requestId) ?? 1) * entry.rating_milli;

  // Improvement passes: apply the single best improving reassignment
  // or pairwise swap, then look again, until nothing improves.
  for (let guard = 0; guard < 500; guard++) {
    let bestGain = 0;
    let move: (() => void) | null = null;

    // Single reassignments into a provider with cap to spare.
    for (const item of workload) {
      const current = assignment.get(item.id);
      if (current === undefined) continue;
      for (const alt of options.get(item.id) ?? []) {
        if (alt.provider === current.provider) continue;
        if (usedCount(alt.provider) >= (caps.get(alt.provider) ?? 0)) continue;
        const gain =
          valueOf(item.id, { provider: alt.provider, model: alt.model, rating_milli: alt.rating_milli }) -
          valueOf(item.id, current);
        if (gain > bestGain) {
          bestGain = gain;
          const target = { provider: alt.provider, model: alt.model, rating_milli: alt.rating_milli };
          move = () => {
            used.set(current.provider, usedCount(current.provider) - 1);
            used.set(target.provider, usedCount(target.provider) + 1);
            assignment.set(item.id, target);
          };
        }
      }
    }

    // Pairwise swaps of the planned providers of two requests.
    for (let i = 0; i < workload.length; i++) {
      for (let j = i + 1; j < workload.length; j++) {
        const a = workload[i];
        const b = workload[j];
        const curA = assignment.get(a.id);
        const curB = assignment.get(b.id);
        if (curA === undefined || curB === undefined) continue;
        if (curA.provider === curB.provider) continue;
        const aAtB = bestOf(a.id, curB.provider);
        const bAtA = bestOf(b.id, curA.provider);
        if (aAtB === null || bAtA === null) continue;
        const gain =
          valueOf(a.id, { provider: aAtB.provider, model: aAtB.model, rating_milli: aAtB.rating_milli }) -
          valueOf(a.id, curA) +
          (valueOf(b.id, { provider: bAtA.provider, model: bAtA.model, rating_milli: bAtA.rating_milli }) -
            valueOf(b.id, curB));
        if (gain > bestGain) {
          bestGain = gain;
          const newA = { provider: aAtB.provider, model: aAtB.model, rating_milli: aAtB.rating_milli };
          const newB = { provider: bAtA.provider, model: bAtA.model, rating_milli: bAtA.rating_milli };
          move = () => {
            assignment.set(a.id, newA);
            assignment.set(b.id, newB);
          };
        }
      }
    }

    if (move === null) break;
    move();
  }

  return assignment;
}

// -------------------------------------------------------- serving order

// The planned provider's candidates first (rating order), then every
// other provider's candidates in plain ranking order. Without a plan
// entry for the request this is just the ranking.
export function servingOrder(
  ranked: Candidate[],
  plan: Plan | null,
  requestId: string | null,
): Candidate[] {
  if (plan === null || requestId === null) return ranked;
  const entry = plan.get(requestId);
  if (entry === undefined) return ranked;
  const head = ranked.filter((c) => c.provider === entry.provider);
  if (head.length === 0) return ranked;
  const tail = ranked.filter((c) => c.provider !== entry.provider);
  return [...head, ...tail];
}
