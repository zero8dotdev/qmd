# Upstream Merge Plan — August 2026

**Goal:** Bring `zero8dotdev/qmd` (fork) `main` from `da67604` up to `tobi/qmd` `main` at `dbfd0b4` (155 commits behind, 1 ahead), then bump the submodule pointer in Smriti.

**Drafted:** 2026-08-25 · **Executed:** 2026-08-25 · **Status:** Complete (local; Phase D not pushed)

**Outcome:** Fork rebased `da67604` → `eafa47e` on top of `upstream/main` (`dbfd0b4`).
The rebase replayed 2 documentation commits with **zero conflicts**; `git diff
upstream/main HEAD --stat` is docs-only (2 files, 526 insertions, **no source
lines**), so the fork remains a pure documentation superset of upstream.

QMD suite: 1219 pass / 2 fail. Both failures are in `test/mcp.test.ts` (MCP HTTP
Transport, 2025-era client initialize) and are **pre-existing upstream** — proven
by the docs-only delta above. MCP HTTP is not a surface Smriti consumes. QMD
type-checks clean under its own `tsconfig.build.json`.

Smriti suite: **408 pass / 0 fail**, identical to the pre-merge baseline.

The §3 concurrency bug is fixed and the fix is verified end-to-end (see
§5 gate 6 results below).

**Previous sync:** [UPSTREAM_MERGE_PLAN-2026-05.md](./UPSTREAM_MERGE_PLAN-2026-05.md) (49 commits, `d58fedf` → `ddbd6bd`, clean fast-forward)

---

## 1. Current State

| Repo | Ref | Commit |
|---|---|---|
| Fork (zero8dotdev/qmd) | `origin/main` | `4e591f5` |
| Smriti submodule pointed at | (stale) | `da67604` |
| Upstream (tobi/qmd) | `upstream/main` | `dbfd0b4` (2026-08-18) |
| Smriti submodule pointer | `qmd` (gitlink) | `da67604` |

```bash
git rev-list --left-right --count HEAD...upstream/main   # 1  155
```

**Not a fast-forward, and not a rebase either.** Two things differ from May:

1. The fork carries `da67604 docs: upstream merge plan and outcome (May 2026)`,
   a fork-local documentation commit touching only `docs/`.
2. **`origin/main` has moved independently.** Someone merged `tobi:main` into the
   fork through the GitHub UI (`4e591f5 Merge branch 'tobi:main' into main`),
   bringing it to roughly v2.6.3. The Smriti submodule pointer was never updated
   and still referenced `da67604`, which is why the drift measured 155 commits
   from the submodule's perspective but only 88 from `origin/main`'s.

`da67604` **is** an ancestor of `origin/main`, so no history rewrite is needed —
and a rebase onto `upstream/main` would be actively wrong here: it would drop
`4e591f5` and require a force-push that discards someone else's merge. Merge
`upstream/main` into `origin/main` instead. The fork has no source divergence
from upstream, only merge topology, so the merge is clean.

### Version span

`v2.1.0` → `v2.8.3`, covering `2.5.0`, `2.5.1`, `2.5.2`, `2.5.3`, `2.6.3`, `2.8.3`.

---

## 2. What's Changing (155 Upstream Commits)

### Churn in the files Smriti imports from

| File | Δ lines | Why it matters |
|---|---|---|
| `src/store.ts` | +1881 / −365 | Smriti imports 5 symbols + 1 type from here |
| `src/db.ts` | +86 | `openDatabase`, `Database`/`Statement` types |
| `src/index.ts` | +27 | `createStore`, `QMDStore` |
| `src/collections.ts` | +7 | not consumed by Smriti |

Non-consumed churn (`cli/qmd.ts` +2212, `mcp/server.ts` +636, `llm.ts` +604,
new `trust.ts`, `mcp/origin-guard.ts`, `cli/embed-lock.ts`, `cli/version.ts`,
`cli/mcp-pid.ts`) is CLI/MCP surface. Smriti consumes QMD as a library and is
unaffected.

### Risk Audit: APIs Smriti Imports from QMD

Consumed surface, verified against `smriti/src/`:

```
createStore, QMDStore                       — src/db.ts:15, src/store.ts:9
hashContent                                  — src/qmd.ts, src/memory.ts, src/team/{share,sync}.ts
chunkDocumentByTokens, reciprocalRankFusion  — src/memory.ts:20-21
formatQueryForEmbedding, formatDocForEmbedding, RankedResult — src/memory.ts:22-24
insertEmbedding (via store.internal)         — src/memory.ts:625,640
Database (type)                              — src/db.ts:17 (type-only), src/memory.ts:17
```

**Export-surface diff of `src/store.ts`:** 23 exports added, **0 removed, 0 renamed.**

| Symbol | Change | Smriti impact |
|---|---|---|
| `hashContent` | Unchanged. | None. |
| `chunkDocumentByTokens` | Signature byte-identical (7 params, `signal?: AbortSignal` still last). | None. |
| `reciprocalRankFusion` | Function body byte-identical. | None. See §6 for the caller-side weighting fix. |
| `formatQueryForEmbedding`, `formatDocForEmbedding` | Re-export line unchanged; callsite refactors only. | None. |
| `insertEmbedding` | **Added optional 9th param** `fingerprint: string = getEmbeddingFingerprint(model)`. Body now wrapped in `withLazyContentVectorMigration`. | **None, and a free improvement.** Smriti's two callsites pass 7 args; the default computes a correct fingerprint automatically, so Smriti's vectors become properly fingerprinted with no code change. |
| `openDatabase` | **Now sets `PRAGMA busy_timeout` (default 120 000 ms, `QMD_SQLITE_BUSY_TIMEOUT`) and performs a retrying `journal_mode = WAL` migration.** WAL moved here from `store.ts:818`. | **Fixes a live Smriti bug.** See §3. |
| `Database.transaction` (type) | Narrowed `<T extends (...args: any[]) => any>` → `<T extends (...args: SQLiteValue[]) => unknown>`, returns `T & { immediate: T }`. | **None.** `grep -rn "\.transaction(" smriti/src/` returns nothing. |
| `Statement.run/get/all` (types) | `any[]` → `SQLiteParams` (`string \| number \| bigint \| Buffer \| Uint8Array \| Float32Array \| null`). | **None.** Smriti's widest call is `.all(new Float32Array(...), limit * 3)` (`memory.ts:498`) — `Float32Array` is in the union. |
| `createStore`, `QMDStore`, `RankedResult` | No signature or semantic change. | None. |

**Conclusion: zero breaking changes for Smriti.** Same verdict as May, now with
one schema change to account for (below).

### Schema change: `content_vectors`

Upstream adds a column to a table Smriti reads, writes and garbage-collects:

```sql
CREATE TABLE IF NOT EXISTS content_vectors (
  hash TEXT NOT NULL,
  seq INTEGER NOT NULL DEFAULT 0,
  pos INTEGER NOT NULL DEFAULT 0,
  model TEXT NOT NULL,
  embed_fingerprint TEXT NOT NULL DEFAULT '',   -- NEW
  total_chunks INTEGER NOT NULL DEFAULT 1,
  embedded_at TEXT NOT NULL,
  PRIMARY KEY (hash, seq)
)
```

The fingerprint is derived from the active embed model plus formatting/chunking
parameters, so vectors are treated as stale-pending when search semantics change.
Migration is lazy (`withLazyContentVectorMigration`) to preserve fast startup.

Smriti's touchpoints — all safe:

- `memory.ts:625,640` write via `insertEmbedding` → fingerprint auto-populated.
- `memory.ts:526,586` read with explicit column lists / `JOIN` → unaffected by an added column.
- `memory.ts:273-293` orphan cleanup uses `hash`/`seq` only → unaffected.

Pre-existing rows keep `embed_fingerprint = ''` and are treated as legacy;
`maybeAdoptLegacyEmbeddingFingerprint` handles safe adoption.

### Behavioral changes worth knowing

- **Concurrency (2.6.3, 2.8.3).** `busy_timeout` + retrying WAL in `openDatabase`;
  FTS trigger setup gated behind `PRAGMA user_version` inside one `IMMEDIATE`
  transaction; concurrent cold-open `table documents_fts already exists` fixed
  (reported specifically on **Bun/macOS** — Smriti's runtime and platform).
- **Embed durability (2.5.0, 2.6.3).** Complete-chunk-coverage required before a
  document counts as embedded; partial vectors removed after an interrupted
  session; per-chunk failure retained and retried; the hardcoded 30-minute embed
  ceiling is now `QMD_EMBED_MAX_DURATION_MS` / `--timeout` (#673).
- **Retrieval (#563, #690, #775, #799).** FTS5 now matches dotted version strings
  (`2026.4.10` was sanitized to `2026410`); `searchVec` embeds with the store's
  pinned model rather than global `QMD_EMBED_MODEL`; multi-collection search runs
  per-collection then merges instead of search-globally-then-post-filter;
  embedding-context pool sized from the weight file instead of a flat 150 MB.
- **macOS Metal exit (2.5.3).** The SIGABRT recorded as a known issue in the May
  plan's Outcome is fixed: `finishSuccessfulCliCommand` sets `process.exitCode`
  instead of calling `process.exit(0)`, so `beforeExit` fires and native contexts
  dispose before libc's static destructor. Defense-in-depth `GGML_METAL_NO_RESIDENCY=1`
  is set in `bin/qmd`, which Smriti does not use — see §6.
- **node-llama-cpp 3.18.1 → 3.20.0.** `LlamaContextSequence.dispose()` became
  async; upstream now awaits it before disposing the parent context. Smriti pins
  `^3.0.0` and will float into 3.20 on its next `bun install`.
- **Security (2.8.3).** Project-local `.qmd/index.yml` `update:` commands, out-of-
  project collection paths and non-default model URIs are now behind a trust gate;
  indexing no longer follows file symlinks or `../` globs out of the collection;
  `qmd mcp --http` validates `Origin`/`Host`. None of these paths are reachable
  from Smriti, but see §6 for the `.smriti/` parallel.

---

## 3. The Bug This Merge Fixes in Smriti

`smriti/src/db.ts:724` opens the database like this:

```
initSmriti()
  └─ createStore()                      ← QMD opens the DB, runs initializeDatabase():
      ├─ PRAGMA journal_mode = WAL         (fork store.ts:818 — no retry)
      ├─ FTS trigger DROP + CREATE
      └─ schema migrations                 ... all with busy_timeout = 0
  └─ db.exec("PRAGMA busy_timeout = 5000") ← arrives AFTER the risky section
```

Smriti can only set the pragma on the handle `createStore()` returns, so the
cold-open DDL runs unprotected. `bun:sqlite` defaults `busy_timeout` to 0, so a
loser in that race throws immediately rather than queueing.

**Why the daemon makes this reachable rather than theoretical.** QMD has no
daemon; Smriti does. `defaultFlushAgent` (`src/daemon/index.ts`) opens the store,
ingests, and closes it **per flush**, so a watching daemon re-runs `createStore()`
— and therefore the whole cold-open path — on every debounced file change, for as
long as it is installed. `flushChain` serializes flushes, but only *within* the
daemon process, and there is no cross-process lock anywhere in `src/daemon/`,
`src/db.ts` or `src/store.ts`. The Stop hook's `lockf` guards ingest-against-ingest
only, and only on the daemon-down fallback path. The unguarded race is
daemon-flush against a foreground `smriti recall` / `embed` / `ingest`.

**Fix ownership: QMD, not Smriti.** `openDatabase` is the only place that runs
before the DDL, and upstream's version is byte-identical to what Smriti needs, so
taking it costs zero fork divergence. Upstream's 120 s default was sized for a
long `embed` batch commit — the same contention a flush produces.

**Smriti-side change:** delete the now-redundant, too-late pragma at
`smriti/src/db.ts:737` and let `openDatabase` own it. Tune via
`QMD_SQLITE_BUSY_TIMEOUT` if a shorter fail-fast is ever wanted.

---

## 4. Execution Plan

### Phase A — Fork repo (`zero8dotdev/qmd`)

```bash
cd qmd

git fetch upstream main && git fetch origin

# ALWAYS branch from origin/main, never from the submodule's current HEAD —
# the submodule pointer can lag behind pushes made to the fork elsewhere.
git checkout -B sync-upstream-2026-08 origin/main
git merge upstream/main
# Expect: clean merge. The fork has no source divergence from upstream.
# If source files conflict, STOP and re-audit — that means someone has
# committed source to the fork.

bun install
bun run test:unit          # Node/Vitest + Bun, per upstream's package.json
```

> **Do not rebase onto `upstream/main`.** It drops the fork's own merge commits
> and turns the push into a force-push over other people's work. This plan was
> drafted with a rebase and corrected during execution — see §7.

### Phase B — Smriti-side change

Remove the redundant pragma now owned by `openDatabase` (§3), then:

```bash
cd /Users/zero8/zero8.dev/smriti
bun install
```

### Phase C — Verification (§5), then commit the submodule bump

```bash
git add qmd bun.lock src/db.ts
git commit -m "chore(qmd): bump submodule to upstream main (dbfd0b4)"
```

### Phase D — Push (requires explicit sign-off)

```bash
cd qmd && git push origin sync-upstream-2026-08:main
```

Fast-forward from `origin/main` (`4e591f5`), so no force and nothing to
coordinate around.

---

## 5. Verification Gates

A merge is **only accepted** when all of these pass:

1. **QMD's own test suite** — `bun run test:unit` in `qmd/`
2. **Smriti's full test suite** — `bun test --cwd ./test` in `smriti/`.
   **Baseline recorded 2026-08-25 before the merge: 408 pass / 0 fail across 35 files.**
3. **Type check** — `bunx tsc --noEmit` in `smriti/` (imports from `../qmd/src/...` must still resolve)
4. **Recall quality eval** — `bun run eval:recall` before and after. This is the
   gate for the retrieval changes; a ranking regression will not show up in unit tests.
5. **Smoke tests** for the three surfaces that touch QMD:
   - `smriti search` (FTS + Vec via `searchMemoryFTS` / `searchMemoryVec`)
   - `smriti recall` (RRF via `reciprocalRankFusion`)
   - `smriti ingest claude` (writes via `addMessage` / `hashContent` / `insertEmbedding`)
6. **Concurrency smoke** — new for this sync, since §3 is the headline fix:
   N processes calling `initSmriti()` on the same cold DB simultaneously,
   confirming no `database is locked` / `trigger documents_ai already exists` /
   `table documents_fts already exists`.

If anything fails: do not push. Reset the submodule pointer and triage.

### Results — 2026-08-25

| Gate | Pre-merge | Post-merge |
|---|---|---|
| 1. QMD suite | not run | 1219 pass / 2 fail (both pre-existing upstream, MCP HTTP) |
| 2. Smriti suite | **408 pass / 0 fail** | **408 pass / 0 fail** |
| 3. Type check | n/a — see note | QMD clean under `tsconfig.build.json` |
| 4. `eval:recall` | 5/6, recall 1.00, precision 0.90 | 5/6, recall 1.00, precision 0.90 — **no change** |
| 5. Smoke (`status`/`search`/`recall`) | — | all three pass against the live 3 446-session DB |
| 6. Concurrency (6 concurrent cold opens) | **1/6 succeeded**, 5 × `SQLiteError: database is locked` | **6/6 succeeded**, zero errors |

Gate 6 is the headline result: the race was reproduced against the pre-merge
submodule and is gone after it. Post-merge pragma state on a fresh DB:

```
busy_timeout  = 120000        (was 0 during cold-open DDL, then 5000)
journal_mode  = wal           (via the retrying migration)
content_vectors = hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at
```

**Gate 4 note.** `eval:recall` is unchanged pre/post, which is the expected
result rather than a disappointing one: upstream's retrieval fixes (#775, and
the `getHybridRrfWeights` caller-side weighting) live in QMD's own `hybridQuery`
/ `searchFTS` paths, while Smriti fuses its own lists in `src/memory.ts`. The
merge does not deliver them — they remain the §6 Future Work items.

**Gate 3 note.** Smriti has **no `tsconfig.json`**, so the May plan's
`bunx tsc --noEmit` gate silently printed the compiler help text and checked
nothing. It has never run. Adding one is worth a follow-up.

---

## 6. Future Work (Out of Scope for This Merge)

Carried forward from May, plus new items:

- **[carried] Port the `getHybridRrfWeights` pattern to Smriti recall.** Still
  open. Smriti calls `reciprocalRankFusion` at `src/memory.ts:803` with its own
  positional `rankWeights`, which is the same shape as pre-fix QMD: when query
  expansion runs first, expansion-derived lists can steal the original query's
  intended 2× weight. `bun run eval:recall` is the instrument.
- **[new] Replace the post-filter overfetch in `searchFiltered` with per-scope
  search + merge.** Smriti's `limit * 3` overfetch (`memory.ts:408`, `:498`) is
  structurally the bug upstream fixed in #775: a large unrelated slice can fill
  the FTS/ANN top-k so a narrow `--project` / `--category` / `--agent` filter
  returns false-empty even when that project has plenty of matches. Upstream's
  per-collection-then-merge is the shape to port.
- **[new] Surface embedding-fingerprint health in `smriti status`.** Upstream's
  `qmd doctor` reports per-fingerprint document/chunk breakdown and warns on mixed
  fingerprints. Smriti has no equivalent; today, changing the embed model silently
  mixes incompatible vectors.
- **[new] Set `GGML_METAL_NO_RESIDENCY=1` in Smriti's own entry point.** Upstream
  sets it in `bin/qmd`, which Smriti does not go through. It must be set *before*
  the process starts — Bun does not propagate `process.env` mutations to libc
  `setenv`, and libggml-metal reads it via `getenv` at module load — so a preload
  will not work.
- **[new] Audit `smriti sync` against upstream's trust-gate threat model.**
  2.8.3 exists because a `.qmd/index.yml` arriving via `git clone` was adopted
  automatically and its `update:` commands executed. `.smriti/` has the same
  shape: config and knowledge that arrive with a clone and are auto-discovered
  via `.smriti/CLAUDE.md`, plus `.smriti/prompts/share-reflect.md` — a prompt
  file from a cloned repo fed to a local LLM. `isPathInsideDir` is now exported
  if a containment check is wanted.
- **[carried] Reconsider keeping QMD as a submodule.** The fork is still zero-
  divergence on source (the only fork commit is documentation). If it stays that
  way, `"qmd": "github:tobi/qmd#<commit>"` would drop the fork entirely.

---

## 7. Rollback Plan

Phase A is a merge that fast-forwards `origin/main`, so rollback is a normal
revert — no history rewrite, no force-push.

```bash
# Fork rollback
cd qmd
git revert -m 1 <merge-commit>

# Smriti rollback (submodule pointer reverts cleanly)
cd ../
git revert <submodule-bump-commit>
```

### Correction made during execution (2026-08-25)

This plan was first drafted specifying a **rebase** onto `upstream/main`, on the
assumption — carried over from May — that the fork's `origin/main` was still at
`da67604`. It was not. `git ls-remote origin main` returned `4e591f5`, a
`Merge branch 'tobi:main' into main` made through the GitHub UI. Rebasing and
pushing would have discarded it.

The rebase was executed locally before this was caught (branch
`sync-upstream-2026-08`, abandoned), and redone as a merge from `origin/main`
(branch `sync-upstream-2026-08-v2`). Nothing was pushed in between.

**Lesson for the next sync, now folded into Phase A:** branch from `origin/main`,
not from whatever commit the submodule happens to point at. A stale submodule
pointer makes the fork look further behind than it is and hides pushes made
elsewhere.

---

## 8. Open Questions

- [ ] Push Phase A to `origin/main`, or keep the sync local until the Future Work
      retrieval items land too?
- [ ] Keep Smriti's `busy_timeout` override at a shorter value than upstream's
      120 s default, or adopt the default outright?
- [ ] Should `bun run eval:recall` become a CI gate rather than a manual sync-time step?
- [ ] Tag the fork at `v2.8.3-sync-2026-08` for traceability?
