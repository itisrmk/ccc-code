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
| `scripts/live_test.py` | Real Claude Code through `ccc` under strict thinking enforcement | About $0.02 on Haiku |

Build first with `cargo build --release -p ccc`. `CCC_LIVE_MODEL=claude-opus-5-5` runs the live tests on another model; it costs about $0.20 per scenario for the first cache write.

Live scenarios (`scripts/live_test.py --list`):

- `headless`: prune rules fire on a real task and the model still finds the failure
- `redact`: a key stays local, the real value comes back, and a resume stays valid
- `enforcement`: control proving an edit to sent history is rejected
- `budget`: the ledger matches Claude Code's reported cost, and a notice is shown then stripped
- `downshift`: an explicit budget downshift changes the model without a history error
- `interactive`: the full-screen TUI in a real terminal via tmux

## Log

Newest first. Record what was tested, what happened and any real finding. Numbers come from commands run, never estimates.

### 2026-10-30

- Phase 4: `ccc stats` ledger, route rules, budget notice, budget downshift, local model route. Live runs found two bugs the unit tests missed. The final `message_delta` repeats cache writes without their lifetimes, which double-counted them and overstated cost by 56%. A notice at the end of a response replaced the answer in `claude -p`, which reports only the last text block, so it now goes first. Local route tested against a fake server only; Ollama isn't installed.
- Measured: Claude Code already sends background title calls to Haiku, so the background route rarely saves anything and stays off by default. A `max_tokens: 1` startup `quota` probe can get a 429 from the API with no edits by `ccc`; the live tests tolerate that one probe only.
- Goal set: finish Phases 4-8 with automated tests, logged here.
- Gate 1 result: token target missed. Rule pruning cut 0.6% of context and 0.4% of cost on real transcripts; simulated server-side clearing cut 11% and 4%. Continuing at the user's direction.
- Phase 3: a live run lost a task's answer to pruning. The `generic` rule dropped the only failure line and a duplicate stub pointed at the trimmed copy. Fixed with regression tests. The bench and all tests had passed before this, so live testing stays mandatory.
- Phase 0-3: 139 tests passing, 0 append-only violations on 2,247 replayed calls, proxy overhead about 0.6 ms at p95.
