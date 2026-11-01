# ccc

Tools for working out what Claude Code really costs and whether your setup really helps.

- **`ccc whatif`** shows where your sessions' money went and what restarting or refreshing them would have changed.
- **`ccc eval`** tests a `CLAUDE.md`, skill or hook on tasks from a repository's history, scored by running the tests.
- **`--advise`** warns you when a session's cache has expired or its context has grown large, before you pay for it.
- **`ccc verify`** checks whether a gateway is safe to put in front of Claude Code.
- **`ccc run`** is a local proxy that redacts secrets, trims noisy tool output and records costs.

None of these except `ccc run` sits between you and the API, unless you point `ccc verify` at a proxy.

**The proxy doesn't deliver the large token savings it was designed for.** On the author's sessions, trimming cuts context by about 0.6% and cost by about 0.4%. The [results](#results) explain why, and [the write-up](docs/token-savings-writeup.md) goes further. `ccc bench` and `ccc whatif` measure your own sessions.

## Read this before using a subscription login

`ccc` forwards whatever credentials Claude Code sends, including a Pro or Max login, and never stores or logs them. Claude Code's docs describe using `ANTHROPIC_BASE_URL` with a subscription login. Anthropic's [legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance) also says third-party developers may not "route requests through Free, Pro, Max, or Ultra plan credentials on behalf of their users" or "intermediate Claude.ai credentials or session tokens". Read that page and decide for yourself, especially before offering `ccc` to anyone else. An API key avoids the question.

## Install

Tested with Rust 1.99 on macOS.

```sh
cargo install --path crates/ccc
```

Optional features, which add native code and need model downloads:

```sh
cargo install --path crates/ccc --features local-model,ner
scripts/fetch-models.sh classifier   # Qwen3-4B, about 2.5 GB
scripts/fetch-models.sh ner          # GLiNER, about 190 MB
```

## Use

```sh
ccc run -- claude                       # Claude Code through ccc
ccc run --stats --budget-usd 5 -- claude
ccc stats                               # what recent sessions cost
ccc dashboard                           # the same, as a page on localhost
ccc bench ~/.claude/projects            # measure effects on your own sessions, offline
ccc whatif ~/.claude/projects           # where the money went, and what would change it
```

### See where the money went

`ccc whatif` replays recorded token counts, so it needs no model and sends nothing anywhere. It breaks spending down by type and context size, then simulates restarting long sessions and keeping caches warm. With no change, the simulation reproduces recorded usage exactly. A restart's figure is an upper limit, because a replay can't see rework or whether the work would have gone as well. `--export FILE` writes totals only, to [share if you choose](docs/share-your-numbers.md).

### Get warned before you pay

```sh
ccc hook-settings --advise > ccc-hooks.json
claude --settings ccc-hooks.json
```

This warns you once when you resume a session whose cache has expired, with the price of the rewrite, and once each as context passes 300K and 600K tokens. Warnings are shown to you and never reach the model. Add `--advise-block` to hold back the first prompt after an expiry, so you can `/compact` first. Sending it again goes ahead. It stays quiet below 100K tokens, which `--advise-min-context` changes.

### Test a setup

```sh
ccc eval mine ./my-repo --test "pytest -q" --out tasks.json    # free, but runs your tests
ccc eval run tasks.json --setup plain --setup mine=./my-setup \
    --out results.jsonl --max-usd 5                            # spends Claude quota
ccc eval report results.jsonl                                  # free
```

A task is a real commit with its source change undone and its tests kept. `mine` keeps only commits whose tests fail without the change and pass with it. A setup is a directory of files placed over the repository, such as a `CLAUDE.md`. The first setup is the baseline.

Each run gets a new git history, so it can't read the answer from later commits, and test files go back before scoring, so editing them doesn't help. Only project settings apply, so your own `CLAUDE.md`, hooks and MCP servers can't affect a run. `--max-usd` stops new runs at a spending limit, and running again continues where it stopped.

Each task counts once however often it runs, and you need at least 7 for a difference to count at all. Agents run with Bash allowed, in a temporary directory but not a sandbox, so only use repositories you trust.

### Check a gateway

```sh
ccc verify --gateway http://127.0.0.1:4000 --spawn "my-gateway --upstream {upstream} --port 4000"
ccc verify --gateway http://127.0.0.1:4000 --upstream-port 9999   # one already running
```

Your gateway forwards to a recording stand-in for the API, so this needs no API key and sends nothing anywhere. It checks that thinking blocks and signatures survive, that history never changes on later turns, that `system` and `tools` stay stable, that repeated requests forward the same bytes, and that streaming, errors and large requests come through. The exit status is non-zero if anything fails.

The stand-in compares bytes and doesn't validate signatures. For the real server's verdict, chain your gateway to `ccc serve --enforce-thinking-check` and run Claude Code through it.

State lives in `~/.ccc`: a redaction key, per-conversation settings, the cost ledger, and anything the optional features save.

## What each stage does

| Stage | Default | What it does |
|---|---|---|
| Redact | **on** | Swaps API keys, private keys, JWTs, connection string passwords, random-looking assignments and emails for placeholders, and restores real values in responses. Add your own patterns in `.ccc/redact-rules`, one regular expression per line. If it fails, the request isn't sent. |
| Prune | **on** | Shortens build logs, long listings and other long command output, and replaces exact repeats with a pointer. Source being read stays whole. Drops lines but never rewrites them, and marks what it dropped. |
| Budget notice | with `--budget-usd` | Shows one line in your terminal when a session passes the budget. The model never sees it. |
| Route | off | `--background-model` moves one-shot calls down. `--budget-downshift` moves a session down once it's over budget. Never moves up. |
| Enrich | off | `--enrich` adds up to three likely files to each prompt. Run `ccc learn ~/.claude/projects` first for better results. |
| Learned router | off | `--learned-router` starts easy conversations on a cheaper model. Train it with `ccc router-train`. |
| Local model | off | `--classify-model` labels turns. `--prune-model` lets a model choose lines to keep from very large output. |
| Name redaction | off | `--ner-model` also hides multi-word names of people and organizations. |

Other commands: `ccc serve`, `ccc index`, `ccc enrich-eval`, `ccc model-bench`, `ccc ner-bench`, and `ccc hook-settings` for hooks through `claude --settings`.

### Team policy

A `.ccc/policy.toml` refuses runs that break its rules. An unknown key is an error, so a typo can't silently turn a rule off.

```toml
require_redact = true
forbid_dump = true
max_budget_usd = 20
allowed_upstreams = ["https://api.anthropic.com"]
redact_rules = ["Project Bluebird"]
```

### Other clients

`ccc run -- gemini` and `ccc run -- codex` point those tools at the proxy, but **their traffic is only forwarded, not redacted or pruned**. With a ChatGPT login, Codex also needs `--upstream https://chatgpt.com/backend-api/codex`. Codex was tested live. Gemini CLI wasn't: only setting its environment variable was.

## Results

Measured on one person's work on an M2 Pro with 16 GB. That's 18 replayed sessions and 2,496 API calls, two of them dominating, in a mode that favours Bash. Yours may differ, so run `ccc bench`.

### Tokens and cost

| Approach | Tool-result tokens | Whole context | Cost |
|---|---|---|---|
| Prune, default limits | -2.6% | -0.6% | -0.4% |
| Prune, tighter limits | -6.1% | -1.4% | -1.1% |
| Server-side clearing, simulated, not shipped | -46.2% | -11.1% | -3.9% |

The original goal was 30%. It isn't reachable this way on these sessions, for these reasons:

- **Caching:** 98.8% of input tokens were already cache reads.
- **What's in context:** tool results are about 22% of it, and most of that is files the model deliberately read, which have to stay whole.
- **Claude Code's own limits:** it already moves output over about 30 KB to a file.
- **History can't be rewritten:** thinking blocks are signed against everything before them, so editing history returns a 400 on newer accounts. Output can only be shortened the first time it's sent.

The server-side clearing row is optimistic, because it ignores the extra calls a model would make to re-read what it lost. Anthropic's docs also warn that context editing is not a savings lever.

The simulation behind these numbers matches recorded usage within 0.1% on context, 0.0% on cache reads, 9.3% on cache writes and 3.6% on cost.

### Where the money goes

From `ccc whatif` on 3,489 calls in 10 sessions, $673.88 at list prices. Two sessions make up 89% of it.

| Where it went | Share of spending |
|---|---|
| Cache reads | 56% |
| Cache writes | 29% |
| Output | 14% |
| Calls with over 300K tokens of context (overlaps) | 76% |
| Rewriting the cache after an idle gap, 26 times (overlaps) | 18% |
| Tool output, estimated (overlaps) | 12% |

| Change, simulated | Effect on cost | Catch |
|---|---|---|
| Restart above 300K, restoring 50K | -36% at most | Ignores rework and any effect on the work. Not tested on real tasks |
| Restart above 300K, restoring 100K | -31% at most | The same |
| Refresh the cache for up to 8 hours when idle | -3.0% at best | Needs something running throughout |
| Refresh the cache for up to 48 hours when idle | +4.6% | Most refreshes are wasted |

### Can `ccc eval` tell setups apart?

Tested on 12 tasks in a synthetic repository (`scripts/eval_fixture.py`), 3 repetitions each on Haiku, 180 runs for $0.90. [The results](docs/results/eval-synthetic-haiku.jsonl) and [setups](docs/results/eval-setups) are in the repo.

| Setup against plain | Pass rate | Cost | Turns | Result |
|---|---|---|---|---|
| The same setup again, as a control | 100% vs 100% | +3% | 4.6 vs 4.5 | No false alarm |
| Told to read everything and re-run everything | 100% vs 100% | 2.2 times as much | 27.6 vs 4.5 | Caught, p = 0.0005 |
| Told it can't edit files | 0% vs 100% | -6% | 5.7 vs 4.5 | Caught, p = 0.0005 |
| Told never to run tests | 94.4% vs 100% | -17% | 4.1 vs 4.5 | Lower pass rate not detected |

It catches large effects. It missed the last one because these tasks are too easy: Haiku passed 34 of 36 without being allowed to test. Small effects on hard, real work are unproven, because this hasn't run on a real repository's history to completion. A run that gives up early is cheap, so the report warns when lower cost comes with fewer passes.

### Does it hurt task success?

Real Claude Code fixing small synthetic bugs, scored by running the tests afterward.

| Model and tasks | Without `ccc` | With `ccc` | With `ccc --enrich` |
|---|---|---|---|
| Haiku, standard tasks | 48/48 | 48/48 | 48/48 |
| Haiku, tasks built so pruning could hurt | 23/24 | 23/24 | not run |
| Sonnet, all tasks | 14/15 | 14/15 | not run |

No difference was seen, but 87 runs per side can't establish the original "within 1 point" target, and these tasks are far smaller than real work. This isn't SWE-bench.

A warning from replay: under default limits, 5 of 23 trimmed results in final requests lost an identifier that a later turn used. The recorded model had seen the full output, so the real effect is probably an extra tool call, but it's a real cost. This sample is small.

### Speed

| Step | Measured | Target |
|---|---|---|
| Whole default pipeline, 1 / 4 / 16 MB request, added p95 | 2 / 11 / 59 ms | 150 ms |
| Qwen3-4B labels | about 160 ms, 20/22 intents right | 80 ms |
| Qwen3-0.6B labels | about 100 ms, 5/22 intents right | 80 ms |
| Qwen3-4B choosing lines in about 4K tokens of output | about 14 s | 600 ms |
| GLiNER on about 2,000 characters | 140 ms p50 | 20 ms |
| File hints, hit at 3 suggestions, past sessions + index | 44.8%, 25.6% precision | none set |

This is why the model-based features are off by default. The prune model also turns itself off at startup if it can't meet its time cap.

## Limits

- **Don't resume across setups.** A conversation's history is compiled the way it started. Resuming it without `ccc`, or without its `~/.ccc` state, resends different history. Claude Code then gets a 400 on newer accounts and recovers by dropping its thinking and rewriting the cache. Older accounts just restart the cache.
- **Settings are pinned per conversation.** `--no-prune` and `--no-redact` affect new conversations only, since switching mid-way would rewrite history. Changing redaction rules mid-conversation has the same problem.
- **Redaction covers the Messages API only.** Other paths are forwarded unchanged. It finds known formats and patterns, not every secret, so it isn't a guarantee.
- **`--dump` captures real prompts and secrets**, with credentials removed. Files are readable only by you.
- **Macs only so far.** Only macOS on Apple Silicon has been tested. The code currently assumes a Unix system.
- **Dollar figures are list prices** from Claude Code's bundled docs, dated 2026-10-06, and notional on a subscription.
- **`~/.ccc` isn't cleaned up.** Its small per-conversation files accumulate.
- **Ollama is untested.** `--local-url` was only tested against a fake server.
- **`ccc verify` hasn't met a real third-party gateway.** It's been run on `ccc` itself and on 13 deliberately broken test gateways.

## Testing

See `CLAUDE.md` for the testing process and a log of findings.

## License

MIT
