# ccc

A local proxy between Claude Code and the Anthropic API. It redacts secrets before they leave your machine, trims noisy tool output, records what each session costs, and can add file hints or route calls between models.

**It doesn't deliver the large token savings it was designed for.** On the author's sessions, trimming cuts context by about 0.6% and cost by about 0.4%. Its measurable value today is redaction and cost visibility. The [results](#results) explain why, and `ccc bench` measures your own sessions.

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
```

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

Measured on one person's work on an M2 Pro with 16 GB. That's 16 replayed sessions and 2,435 API calls, two of them dominating, in a mode that favours Bash. Yours may differ, so run `ccc bench`.

### Tokens and cost

| Approach | Tool-result tokens | Whole context | Cost |
|---|---|---|---|
| Prune, default limits | -2.6% | -0.6% | -0.4% |
| Prune, tighter limits | -6.1% | -1.4% | -1.1% |
| Server-side clearing, simulated, not shipped | -46.4% | -11.3% | -3.9% |

The original goal was 30%. It isn't reachable this way on these sessions, for these reasons:

- **Caching:** 98.8% of input tokens were already cache reads.
- **What's in context:** tool results are about 22% of it, and most of that is files the model deliberately read, which have to stay whole.
- **Claude Code's own limits:** it already moves output over about 30 KB to a file.
- **History can't be rewritten:** thinking blocks are signed against everything before them, so editing history returns a 400 on newer accounts. Output can only be shortened the first time it's sent.

The server-side clearing row is optimistic, because it ignores the extra calls a model would make to re-read what it lost. Anthropic's docs also warn that context editing is not a savings lever.

The simulation behind these numbers matches recorded usage within 0.1% on context, 0.0% on cache reads and 3.8% on cost. Simulated cache writes are 10.4% high, just over the 10% target.

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
| Whole default pipeline, 1 / 4 / 16 MB request, added p95 | 2 / 11 / 47 ms | 150 ms |
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

## Testing

See `CLAUDE.md` for the testing process and a log of findings.

## License

MIT
