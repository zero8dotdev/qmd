# Upstream Merge Plan — May 2026

**Goal:** Bring `zero8dotdev/qmd` (fork) `main` from `d58fedf` up to `tobi/qmd` `main` at `ddbd6bd` (49 commits behind), then bump the submodule pointer in Smriti.

**Drafted:** 2026-05-18 · **Executed:** 2026-05-18 · **Status:** Complete

**Outcome:** Fork main fast-forwarded from `d58fedf` to `ddbd6bd`. QMD test suite: 868 pass / 0 fail. Stale branches `perf/addmessage-upsert-opt` and `refactor/move-memory-ollama-to-smriti` deleted from origin. Note: QMD process exits with SIGABRT on macOS due to a known Metal device cleanup issue during atexit (not a test regression — all tests pass before exit).

---

## 1. Current State

| Repo | Ref | Commit |
|---|---|---|
| Fork (zero8dotdev/qmd) | `origin/main` | `d58fedf` |
| Upstream (tobi/qmd) | `upstream/main` | `ddbd6bd` |
| Smriti submodule pointer | `qmd` (gitlink) | `d58fedf` |

**Critical finding:** `origin/main` is a **pure ancestor** of `upstream/main` — zero fork-specific commits on main. This is a clean fast-forward at the Git level.

```bash
# Verified:
git log --oneline upstream/main..origin/main   # empty
git merge-base origin/main upstream/main       # = d58fedf
```

### Other fork branches (stale, will be deleted)

| Branch | Status | Reason |
|---|---|---|
| `origin/perf/addmessage-upsert-opt` | Stale | Adds `memory.ts`/`ollama.ts` to QMD — both modules have since moved to `smriti/src/`. Obsolete. |
| `origin/refactor/move-memory-ollama-to-smriti` | Stale | The move it describes is already complete in Smriti. Obsolete. |
| Local `main` (commit `7ec50b8`) | Stale | Predates the modules-to-smriti move. **Diverged from `origin/main` — must be reset, not merged.** |
| Local `sync-upstream` | Use for this work | Repurpose or recreate from `origin/main`. |

---

## 2. What's Changing (49 Upstream Commits)

### Categories

| Type | Count | Examples |
|---|---|---|
| Search/RRF correctness | 3 | `004714a` RRF weighting by query type, `d045a8b` CJK FTS, `5b9f472` embed collection filter |
| Stability / GPU | 6 | `60c75cb` Metal cleanup abort, `e8229d8` Windows CUDA parallelism, `1f75737` GPU status |
| Embedding | 4 | `910ca07` partial embeddings pending, `b59ba6a` cleanup lifecycle, model resolution |
| CLI / MCP UX | 8 | `c18c74a` serve QMD skill from CLI, `9cecdc8` terse MCP collection summary, `e36ab96` HTTP rerank control |
| Path / docid | 4 | `dff6513` preserve docids across case renames, `2dc8634` qmd:/// aliases, `aa1818e` clamp fromLine |
| Bench / test infra | 6 | `2e0c743` local-index bench, Node+Bun matrix, lifecycle regression |
| Routine merges | 18 | First-parent merges of the above |

### Risk Audit: APIs Smriti Imports from QMD

Smriti consumes QMD via the vendored submodule and imports these symbols directly from `../qmd/src/store.ts`, `../qmd/src/index.ts`, `../qmd/src/db.ts`:

```
createStore, QMDStore                       — src/db.ts, src/store.ts
initializeMemoryTables                       — src/qmd.ts (re-export from smriti/src/memory.ts)
hashContent                                  — src/qmd.ts, src/team/share.ts, src/team/sync.ts
chunkDocumentByTokens, reciprocalRankFusion  — src/memory.ts
formatQueryForEmbedding, formatDocForEmbedding, RankedResult — src/memory.ts
insertEmbedding                              — src/memory.ts
Database (type)                              — src/db.ts, src/memory.ts
```

**Diff scan of these symbols across the 49 commits:**

| Symbol | Change | Smriti impact |
|---|---|---|
| `insertEmbedding` | Added optional 7th param `totalChunks?: number` | None — backward compatible. Smriti's callsites continue to work; new behavior (partial-embedding pending state from `910ca07`) is opt-in. |
| `formatQueryForEmbedding` | Callsite refactor only (passes resolved `embedModel` instead of `llm.embedModelName`). Signature unchanged. | None. |
| `reciprocalRankFusion` | Unchanged. **But `hybridQuery` now weights "original"-type lists at 2x via `getHybridRrfWeights` (commit `004714a`).** | **Improvement.** Smriti calls `reciprocalRankFusion` directly with its own weights — unaffected. QMD callers of `hybridQuery` get better rankings. |
| `createStore`, `QMDStore`, `hashContent`, `chunkDocumentByTokens`, `formatDocForEmbedding`, `RankedResult`, `initializeMemoryTables`, `Database` | No signature or semantic change. | None. |

**Conclusion: zero breaking changes for Smriti.**

### Behavioral changes worth knowing

- **CJK FTS migration (`d045a8b`):** First post-merge call into `documents_fts` rebuilds the FTS table to space-separate CJK characters. **One-time, on-disk migration** in `~/.cache/qmd/index.sqlite`. Smriti's recall path uses its own `memory_fts` table (separate from `documents_fts`), so Smriti recall is unaffected. Direct `qmd` CLI users will see a brief one-time delay on first query.
- **RRF weighting fix (`004714a`):** Affects QMD's `hybridQuery` only. Smriti's `searchMemoryFTS` / `searchMemoryVec` / RRF fusion is a separate code path — not directly improved, but the new `getHybridRrfWeights` helper is a pattern worth porting later (see Future Work).
- **Docid stability (`dff6513`):** Case-only renames now preserve docid. Smriti stores docids inside session messages — pre-merge docids remain valid post-merge.
- **Partial embeddings pending (`910ca07`):** Crash mid-embedding-batch now leaves chunks marked pending instead of orphaning the document. Smriti doesn't use QMD's embed batch (uses its own loop), no impact.

---

## 3. Execution Plan

### Phase A — Fork repo (`zero8dotdev/qmd`)

```bash
cd qmd

# 1. Confirm upstream is current
git fetch upstream main
git fetch origin

# 2. Move to a working branch off origin/main
git checkout -B sync-upstream-2026-05 origin/main

# 3. Fast-forward to upstream/main
git merge --ff-only upstream/main
# ↑ MUST succeed without manual conflict resolution. If it fails,
#   re-verify that origin/main has no commits not in upstream/main:
#     git log --oneline upstream/main..origin/main
#   Should be empty. If non-empty, stop and re-plan.

# 4. Run the QMD test suite locally before pushing
bun install
bun test --preload ./src/test-preload.ts test/

# 5. Push to fork main (fast-forward push, no force needed)
git push origin sync-upstream-2026-05:main
```

### Phase B — Cleanup stale branches

```bash
# Delete obsolete fork branches (after confirming nothing depends on them)
git push origin --delete perf/addmessage-upsert-opt
git push origin --delete refactor/move-memory-ollama-to-smriti

# Local cleanup
git branch -D main           # local main is stale (7ec50b8) — recreate
git checkout main            # tracks origin/main, now at ddbd6bd
git branch -D sync-upstream  # if no longer needed
```

### Phase C — Smriti submodule bump

```bash
cd /Users/zero8/zero8.dev/smriti

# 1. Update submodule pointer
git submodule update --remote qmd
# OR explicit:
#   cd qmd && git checkout main && git pull && cd ..

# 2. Reinstall (file: dep resolves the new commit)
bun install

# 3. Run Smriti's full test suite — this is the real gate
bun test

# 4. Smoke-test the CLI surfaces that touch QMD search:
bun src/index.ts recall "test query" --check-conflicts
bun src/index.ts search "test query"
bun src/index.ts status

# 5. Commit the submodule bump
git add qmd bun.lock
git commit -m "chore(qmd): bump submodule to upstream main (ddbd6bd)

Pulls 49 upstream commits including RRF weighting fix (#004714a),
CJK FTS support (#d045a8b), embed collection filter fix (#5b9f472),
and macOS Metal cleanup stability (#60c75cb).

No breaking changes to APIs consumed by Smriti."
```

---

## 4. Verification Gates

A merge is **only accepted** when all of these pass:

1. **QMD's own test suite** — `bun test --preload ./src/test-preload.ts test/` in `qmd/`
2. **Smriti's full test suite** — `bun test` in `smriti/`
3. **Type check** — `bunx tsc --noEmit` in `smriti/` (imports from `../qmd/src/...` must still resolve)
4. **Smoke tests** for the three surfaces that touch QMD:
   - `smriti search` (FTS + Vec via `searchMemoryFTS` / `searchMemoryVec`)
   - `smriti recall` (RRF via `reciprocalRankFusion`)
   - `smriti ingest claude` (writes via `addMessage` / `hashContent` / `insertEmbedding`)
5. **CHANGELOG note** in Smriti's CHANGELOG if it exists

If anything fails: do not push. Revert the submodule bump and triage.

---

## 5. Rollback Plan

The fork main move is a fast-forward push — easy to revert if a regression slips through:

```bash
# Fork rollback (force-push back to d58fedf — only on main if no one else has pulled)
cd qmd
git push origin --force-with-lease d58fedf:main

# Smriti rollback (submodule pointer reverts cleanly)
cd ../
git revert <submodule-bump-commit>
```

`--force-with-lease` is required because we're rewriting a fast-forward; refuses if anyone has pushed since. Coordinate with team before doing this.

---

## 6. Future Work (Out of Scope for This Merge)

These are improvements to consider after the merge lands, not blockers:

- **Port `getHybridRrfWeights` pattern to Smriti recall.** Smriti's RRF fusion uses positional weights similar to the pre-fix QMD logic. Same bug class applies — when query expansion runs first, expansion-derived lists can steal the original-query weight. Worth fixing in `smriti/src/memory.ts` and `src/search/recall.ts`.
- **Reconsider keeping QMD as a submodule vs. upstreaming Smriti's QMD usage as a stable consumer.** The fork has zero divergence — if it stays that way, switch to `"qmd": "github:tobi/qmd#<commit>"` and drop the fork entirely.
- **Wire the `--check-conflicts` recall path through `hybridQuery` instead of separate FTS+Vec+RRF.** Would get the upstream RRF fix for free.

---

## 7. Open Questions

- [ ] Is anything in the codebase still consuming `perf/addmessage-upsert-opt` or `refactor/move-memory-ollama-to-smriti`? (Check before deleting.)
- [ ] Do we want to tag the fork at `v2.1.0-sync-2026-05` for traceability, or just rely on the commit hash in Smriti's submodule pointer?
- [ ] Should the QMD test suite run in CI for this fork, or only at sync time?
