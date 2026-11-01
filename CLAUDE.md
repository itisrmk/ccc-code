# ccc

A local proxy between Claude Code and the Anthropic API that compiles every request. Plan: `~/.claude/plans/resilient-noodling-sun.md`.

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

## Log

Newest first. Record what was tested, what happened and any real finding. Numbers come from commands run, never estimates.

### 2026-11-01

- Final run on this work: `scripts/check.sh` passed with 302 tests, and 19 of 20 live scenarios passed with the 20th, Codex, skipped because it spends ChatGPT quota and no proxy code changed. Full pipeline 2, 11 and 59 ms at 1, 4 and 16 MB.
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
