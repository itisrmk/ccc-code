# ccc

Tools for measuring and reducing what Claude Code costs, including a local proxy that compiles every request.

`cargo` isn't on `PATH` here. Use `export PATH=/opt/homebrew/opt/rustup/bin:$PATH`.

## Rules that must not break

- **Compiling is append-only.** A thinking block's signature binds `system`, `tools` and every earlier message. Editing sent history is a 400 on accounts created on or after 2026-08-31 and restarts the prompt cache on every account. Stage output must be a pure function of the request, and appending messages must not change how earlier ones compile. Anything non-deterministic, such as a model step, must only act on the newest message and persist its decision.
- **Never touch thinking blocks** in requests or responses.
- **Redaction fails closed.** Other stages pass through on failure.
- **No real prompts, code or secrets in the repo.** Fixtures are synthetic, and fake credentials are assembled at run time.

## Automated testing

Run these after every change. Add a line to the log below for each phase or real finding.

| Command | What it does | Cost |
|---|---|---|
| `scripts/check.sh` | Format, clippy with warnings as errors, all tests, docs, proxy overhead | Free, offline |
| `target/release/ccc bench ~/.claude/projects` | Replays real transcripts: token and cost effect per scenario, append-only violations, fidelity, stage latency | Free, offline |
| `target/release/ccc whatif ~/.claude/projects` | Spending breakdown from recorded usage, restart and keep-warm simulations | Free, offline |
| `scripts/live_test.py` | Real Claude Code through `ccc` under strict thinking enforcement | About $0.02 on Haiku |

Build first with `cargo build --release -p ccc --features local-model,ner`, so the model scenarios can run. They skip themselves without the feature or the weights in `~/.ccc/models`, fetched with `scripts/fetch-models.sh`. `ccc model-bench FILE --transcripts DIR` measures a model on this machine. `CCC_LIVE_MODEL=claude-opus-5-5` runs the live tests on another model; it costs about $0.20 per scenario for the first cache write.

`scripts/restart_test.py` measures restarting against one long session, with real runs. `ccc enrich-eval DIR --embedding-model FILE --rerank-model FILE` measures file suggestions on real history, free. Keep experiment output in `target/experiment` or `docs/results`, since the session scratchpad doesn't survive.

`ccc eval` tests Claude Code setups. `scripts/eval_fixture.py DIR` builds a task repository from this project's own synthetic bugs, so no outside code runs: `ccc eval mine DIR --test "python3 run_tests.py" --out tasks.json`, then `ccc eval run`. About $0.005 per run on Haiku.

`scripts/e2e.py` measures task success (Gate 3): real Claude Code fixing small synthetic bugs with and without `ccc`, scored by running the tests afterward. About $0.02 per run on Haiku and $0.15 on Sonnet. `--check-tasks` confirms each task starts out failing, free.

Live scenarios (`scripts/live_test.py --list`):

- `headless`: prune rules fire on a real task and the model still finds the failure
- `redact`: a key stays local, the real value comes back, and a resume stays valid
- `enforcement`: control proving an edit to sent history is rejected
- `budget`: the ledger matches Claude Code's reported cost, and a notice is shown then stripped
- `downshift`: an explicit budget downshift changes the model without a history error
- `enrich`: hints reach the model and stay stable across turns and a resume
- `hooks`: hooks registered through settings fire, build an index and log cost
- `names`: a name only a model can spot stays local and comes back real
- `learned_router`: a trained router moves an easy conversation down and keeps it there
- `policy`: a team policy refuses a run that breaks it, and its rules apply
- `dashboard`: the dashboard serves the ledger's sessions
- `codex`: Codex through the proxy; opt in with `CCC_LIVE_CODEX=1`, since it spends ChatGPT quota
- `local_model`: real weights on the request path, with kept lines verbatim and history valid
- `slow_model_floor`: a model too slow for its cap turns its stage off
- `interactive`: the full-screen TUI in a real terminal via tmux
- `advise`: resuming after the cache expired warns the person and the warning never reaches the model
- `advise_block`: blocking holds back the first prompt after expiry, and sending it again works
- `verify`: `ccc verify` passes `ccc`'s own proxy, free
- `gateway_chain`: Claude Code through a gateway to an enforcing `ccc` to the real API
- `eval`: `ccc eval` mines a commit and scores a real agent by running the tests
- `setup`: a clean install under a throwaway home reports healthy and leaves nothing behind
- `status_line`: the real TUI shows our status line
- `service_recovers`: killing the background service never hurts a prompt
- `trim`: a long log reaches the model shorter through the hook, with its failure line kept
- `rules`: a learned rule stays unused until approved, and a planted instruction is never learned
- `init`: `ccc init` writes hooks that really warn, and removes them again, leaving other settings
- `handoff`: a fresh session given only `ccc handoff` knows what the first session was told
- `enhance`: the prompt hook adds files as context, leaves the person's words alone, and stays valid
- `ask`: `ccc ask` answers from local code with real weights, offline
- `mcp`: Claude really calls the local search tool and uses what it finds

## Log

Newest first. Record what was tested, what happened and any real finding. Numbers come from commands run, never estimates.

### 2026-11-01

- Final run on Phase 2: `scripts/check.sh` passed with 393 tests, and 29 of 30 live scenarios passed with Codex skipped, since it spends ChatGPT quota and no proxy code changed. Full pipeline 3, 14 and 47 ms at 1, 4 and 16 MB. Phase 3, other agents and launch, waits on the user.
- Hiding commands behind `ccc advanced` silently turned three live scenarios into skips, since they looked for their features in `--help`, and a suite reporting all passed hid it. They now read `ccc advanced`. Read the skip list, not just the count.
- Haiku sometimes skips thinking, seen once: `headless` and `enforcement` then fail because there's nothing to enforce. They passed 3 of 3 on rerun. Keep them strict, since a pass with no thinking block proves nothing.
- Phase 2 of the product plan, tested on 7 real tasks from one of the author's Rust projects with `cargo` off the agent's `PATH`, one repetition. Haiku, $1.48: plain 7/7, learned rule 6/7, trim 5/7, rewrite 7/7. Sonnet, $2.76: plain 6/7, learned rule 5/7. No difference reached the bar, and none passed more tasks, so nothing is on by default. One trim run failed after a single turn at above-average cost, and why wasn't investigated. The default-on gate of 20 real tasks can't be met from the author's repositories so far.
- Learned rules on real history: 178 prompts yielded nothing lasting. Asked directly, Gemma 4 E2B kept turning task requests into rules: 1 useful of 3, and few-shot examples made it worse at 15, nearly all noise. A narrow filter on lasting language plus a second yes-or-no question left 0. Commands that kept working were far better, with no model: 9 proposals, including `export PATH=/opt/homebrew/opt/rustup/bin:$PATH` seen 417 times and `scripts/check.sh`. Real command lines are chained, so a filter skipping chains had dropped 1,223 runs until it was changed to read the pieces.
- `PostToolUse` `updatedToolOutput` works on Bash: 7,286 tokens trimmed live, failure kept, with no command rewriting, so permissions are untouched. The plan had assumed `PreToolUse`.
- A commit went out with a failing test, because a script committed regardless of `check.sh`. The failure was real: a process starting at the wrong moment can inherit a dead service's socket, so connecting seemed to work and the next service refused to start. A service must now answer a ping. Commit steps now depend on the checks passing.
- Phase 1 of the product plan: `ccc setup`, `uninstall`, `doctor`, `status`, `gain`, a status line, a background service, an Ollama and LM Studio backend, a plugin and `install.sh`. A clean install under a throwaway home took 28 s with the 2.8 GB model left downloading in the background, against 239 s waiting for it. Service memory 455 MB with embeddings loaded, 2,755 MB with Gemma too. A hook answered in 0.36 s with the service killed. The real TUI showed `week limit 35% · cache 1h0m · context 44K`, from a real session.
- `ccc uninstall` restored settings exactly once a real comparison ruled out a false alarm: `claude mcp add` itself rewrites `"opus"` to `"opus[1m]"`.
- A real redaction limit, seen once in 4 runs of `policy`: Haiku replied with a mangled placeholder, `<be6vzn1v-codename>`, built from the temporary path, which can't be turned back into the real value. The value stayed local, so it fails safe, but the person sees nonsense.
- `v0.1.1` was refused by the permission classifier as a new public surface, so Apple Silicon still has no prebuilt binary. Pushing a tag is the user's to do.
- Published to https://github.com/itisrmk/ccc-code at the user's go-ahead. History scanned first: no private paths or project names, and the only key-shaped strings are fake ones in tests. First CI run passed, and `cargo install --git` from the public repo builds and runs. Not released: no tag, npm package or Homebrew formula yet, so `.github/workflows/release.yml` hasn't run. The user chose not to run downloaded code, so `ccc verify` hasn't met a third-party gateway.
- First `ccc eval` on a real repository, one of the author's own Rust projects: 160 commits gave 7 tasks with limits of 400 lines and 6 files, 2 with 120 lines and 3 files, and 17 commits failed their own tests. Haiku, 21 runs, $0.65: plain 7/7, "never run tests" 1/7 in 2.0 turns against 20.1, enhanced 5/7 at 62% more cost. Sonnet, 14 runs, $2.66: plain 6/7, enhanced 5/7 at 9% less. One repetition each. Enhancement never passed more tasks, so it stays off and the README says so. Pass-rate differences missed the bar because 7 tasks with a tie can't go below p = 0.03. Keep results from private repositories out of this repo: `target/experiment/own` is ignored for that reason.
- Restart test at scale (`scripts/restart_test.py --large`), real runs: Sonnet $3.70 against $5.13 (-28%, 2 runs per arm, peak 271K against 522K, one restart). Haiku $1.76 against $2.15 (-18%, 1 run per arm, peak 530K against 947K). Every run passed 12/12 modules, followed the first prompt's rule, and logged every ticket from notes it had to read. Small runs and large runs disagree because each restart rewrites the cache, so it only pays once context is large. Tasks are independent, which flatters a restart. About $22 at list prices.
- Prompt enhancement on the 12 synthetic tasks: Haiku cost 14% less and took 4.2 turns against 4.7 (p = 0.003, 0.017, over 72 pairs, extended after a first look, so somewhat optimistic). Sonnet cost 14% more and took 4.8 turns against 3.1 (p = 0.0005, 0.001, over 24 pairs). All passed. Same feature, opposite results, so it's off by default. Why Sonnet does worse wasn't investigated.
- File suggestions on 106 real prompts: keywords 20.8% hit at 3, EmbeddingGemma 25.5%, both 28.3%, past sessions plus keywords 45.3%, all three by rank fusion 40.6%, reordered by Gemma 4 E2B 37.7% in 638 ms, by Qwen3-4B 32.1% in 1.19 s. Reordering hurts and embeddings don't improve on past sessions, so enhancement uses vectors only without history and no model chooses files.
- llama-cpp-2 0.1.160, the latest, can't load EmbeddingGemma 2 (`gemma-embedding2`, "null result from llama cpp"), so the first EmbeddingGemma is used, under Gemma terms. Gemma 2 is superseded: Gemma 4 E2B is Apache 2.0 and the bundled llama.cpp loads it. The shared llama backend is now process-wide, since it can only be set up once.
- Restart test, small sessions, real runs, 12 tasks, restarting every 4: Haiku $0.0288 against $0.0304 (-5%, 6 runs per arm), Sonnet $0.4256 against $0.3672 (+16%, 3 runs). 12/12 modules and 12/12 of a rule from the first prompt in every run, so the handoff lost nothing. Peak context was only 55K, far from the 300K where `whatif` simulates savings.
- `claude -p --output-format json` reports a running total on resume. Summing it made a pilot look 60% cheaper ($0.20 against $0.08) when the real gap was 5%. Score cost from recorded usage.
- `ccc eval report` used whichever setup finished first as its baseline, which printed a comparison backward. The baseline is now the first setup given.
- Final run on this work: `scripts/check.sh` passed with 333 tests, and 24 of 25 live scenarios passed with the 25th, Codex, skipped because it spends ChatGPT quota and no proxy code changed. Full pipeline 2, 11 and 46 ms at 1, 4 and 16 MB.
- Added `ccc eval`. On 12 synthetic tasks, 3 repetitions, Haiku, 180 runs, $0.90: an identical copy showed no false alarm. A wasteful `CLAUDE.md` was caught on cost (2.2 times) and turns (27.6 vs 4.5), p = 0.0005. A read-only one was caught on pass rate (0% vs 100%), p = 0.0005. "Never run tests" wasn't caught (94.4% vs 100%), because Haiku solves these tasks without testing. Large effects only; small ones on real work are unproven.
- `ccc eval` called "never run tests" BETTER on cost, because giving up early is cheap. The report now warns when lower cost comes with fewer passes.
- A first `ccc eval` run on real `more-itertools` history left 63 folders in `~/.claude/projects`: Claude Code makes one per working directory even with `--no-session-persistence`. Removed them, and `ccc eval` now removes its own. Its results were lost with the temporary directory when the session ended, and the rerun was refused by the permission classifier because it executes downloaded code. So nothing here is measured on a real repository's history. Keep experiment output somewhere durable.
- Added `ccc whatif`: reads usage only, and reproduces recorded cost exactly with no change. On 3,489 calls, $673.88: over 300K tokens of context is 76% of spending, idle cache rewrites 18%, tool output about 12%. Restarting above 300K while restoring 50K is -36% at most, unproven on real work. Keep-warm nets -3.0% at best and costs more at 48 hours, against an expectation that it would win.
- Added `--advise`: warnings go out as `systemMessage`, since plain stdout from these hooks joins the model's context. Live tests age a throwaway transcript instead of waiting for a cache to expire.
- Added `ccc verify`: 22 checks against a recording stand-in, proven by 13 deliberately broken gateways that each fail or warn on the right check. `ccc` passes. No real third-party gateway tested.
- Correction: an earlier search suggested nothing measures whether a `CLAUDE.md` helps. `mdarena` (HudsonGri/mdarena, 68 stars) and the "Evaluating AGENTS.md" paper (arXiv 2602.11988) do. `ccc eval` isn't unique.
- Bench refreshed after cleanup: 18 sessions, 2,496 calls, cache writes now within target at 9.3%.

### 2026-10-30

- Final run on the finished binary: `scripts/check.sh` passed with 237 tests, all 15 live scenarios passed including Codex and the interactive TUI, and `scripts/e2e.py --reps 1` passed 15/15 in all three configurations. Proxy overhead 0.5 ms at p95; full pipeline 2, 11 and 47 ms at 1, 4 and 16 MB.
- Gate 3, task success, scored by running tests afterward: Haiku standard tasks 48/48 without `ccc`, 48/48 with, 48/48 with `--enrich`. Haiku adversarial tasks, where the key line has no failure keyword, 23/24 and 23/24. Sonnet 14/15 and 14/15. No difference, but 87 runs per side can't establish "within 1 point", and these are small synthetic tasks, not SWE-bench. Treat the gate as not contradicted rather than proven.
- Test runs polluted `~/.claude/projects` with 262 throwaway sessions, which moved the bench's cache-write error from 11.6% to 23%. Removed those and made both harnesses clean up after themselves. Always check `files read` in bench output.
- Phase 8: learned router (68.3% held-out on 60 labelled requests, against 50.0% for guessing; a mechanism on thin data, off by default), team policy, Codex and Gemini forwarding, local dashboard. Codex tested live: it tries a WebSocket, gets a 405 and falls back to HTTP. Gemini CLI isn't installed, so only its environment variable is tested. A hosted dashboard wasn't built, since it was optional and hosting means publishing.
- Phase 7, names: GLiNER takes 140 ms at p50 per 2,000 characters against the spec's 20 ms. On this repo's source it flagged `self`, `name`, `root` and `decisions`, and replacing those would wreck everything the model reads, so only multi-word capitalized names count. Single-word names belong in project rules.
- Phase 7, suggestions: `ccc enrich-eval` on real history, using only what came before each prompt: past sessions hit a touched file at 3 suggestions 42.6% of the time at 28.3% precision, the index alone 20.0% at 11.5%, both 44.8% at 25.6%. Enrich stays opt-in.
- Phase 7, docs check: a `PreCompact` hook can only read or block, so it can't feed a summary as the spec wanted. Learned context goes through `SessionStart`, which also fires with `source: compact`.
- Terms: Claude Code's compliance page says third-party developers may not route requests through Free, Pro, Max or Ultra credentials on behalf of their users, or intermediate those credentials. Affects distributing `ccc`, not building it. See README.
- Gate 2 passed: the full default pipeline adds 2 ms at p95 on a 1 MB request, 11 ms at 4 MB and 46 ms at 16 MB, under the 150 ms limit (`overhead` test, M2 Pro, 16 GB). Other machine classes unmeasured.
- Phase 5 measured on real weights (`ccc model-bench`, 22 hand-labelled requests): Qwen3-0.6B took 101 ms at p50 with 5/22 intents right, so the spec's model isn't accurate enough. Qwen3-4B took 173 ms with 20/22 intents and 18/22 difficulties right, but misses the 80 ms cap. Few-shot examples and prompt prefix caching made that difference; the first attempt had 4/22. Blob pruning takes 14.5 s at p50 against a 600 ms cap (373 tokens/s), so it can't run at the spec's cap on this machine. Both stay off by default; the blob stage also disables itself at startup if the model is too slow for its cap.
- Phase 5 bugs found running real weights: `sample()` already advances the grammar, so a second `accept` crashed; Metal aborts at exit unless the worker thread is joined; an answer cut off at the token limit failed strict parsing and was thrown away.
- Phase 4: `ccc stats` ledger, route rules, budget notice, budget downshift, local model route. Live runs found two bugs the unit tests missed. The final `message_delta` repeats cache writes without their lifetimes, which double-counted them and overstated cost by 56%. A notice at the end of a response replaced the answer in `claude -p`, which reports only the last text block, so it now goes first. Local route tested against a fake server only; Ollama isn't installed.
- Measured: Claude Code already sends background title calls to Haiku, so the background route rarely saves anything and stays off by default. A `max_tokens: 1` startup `quota` probe can get a 429 from the API with no edits by `ccc`; the live tests tolerate that one probe only.
- Goal set: finish Phases 4-8 with automated tests, logged here.
- Gate 1 result: token target missed. Rule pruning cut 0.6% of context and 0.4% of cost on real transcripts; simulated server-side clearing cut 11% and 4%. Continuing at the user's direction.
- Phase 3: a live run lost a task's answer to pruning. The `generic` rule dropped the only failure line and a duplicate stub pointed at the trimmed copy. Fixed with regression tests. The bench and all tests had passed before this, so live testing stays mandatory.
- Phase 0-3: 139 tests passing, 0 append-only violations on 2,247 replayed calls, proxy overhead about 0.6 ms at p95.
