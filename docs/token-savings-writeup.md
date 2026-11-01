# We built a token-saving proxy for Claude Code. It saved 0.4%.

> **Draft, not published.** Every number comes from one developer's sessions, and two of them make up 89% of the spending. It needs numbers from other developers before it says anything general. See [share-your-numbers.md](share-your-numbers.md).

The plan was simple. Put a proxy between Claude Code and the API, shorten the noisy tool output, and cut token use by 30%. We built it, replayed real sessions through it, and got this:

| Approach | Tool output | Whole context | Cost |
|---|---|---|---|
| Prune, default limits | -2.6% | -0.6% | -0.4% |
| Prune, tighter limits | -6.1% | -1.4% | -1.1% |
| Clearing old tool results on the server, simulated | -46.2% | -11.1% | -3.9% |

Even clearing nearly half of all tool output, which is optimistic because it ignores the re-reading that would follow, only saves 3.9%.

The replay is checked against what the API really reported. Across 2,496 calls in 18 sessions it matches recorded context within 0.1%, cache reads within 0.1% and cost within 3.6%.

## Why it didn't work

**Almost everything is already a cache read.** About 99% of input tokens came from the cache, at a small fraction of the normal price. A token you remove from context was costing very little on each call.

**Tool output isn't where the money is.** Writing each tool result to the cache and reading it on every later call adds up to about 12% of spending, by estimate. That's the ceiling for any tool that only shrinks tool output, on these sessions, even if it removed all of it.

This is why a large reduction in output and a small reduction in the bill can both be true. A tool can honestly report that it shrank command output by 80%. That measures output. Ask what it does to the bill.

**Most tool output is deliberate.** Much of it is source files the model chose to read, and they have to stay whole. Claude Code also already moves very long output to a file.

**You can't edit the past.** A thinking block's signature covers everything before it, so changing sent history gets a 400 on accounts created on or after 2026-08-31, and restarts the prompt cache on every account. Output can only be shortened the first time it's sent, which rules out most clever compression.

## Where the money did go

| Where it went | Share of spending |
|---|---|
| Cache reads | 56% |
| Cache writes | 29% |
| Output | 14% |

Cutting across those:

- **Calls with over 300K tokens of context: 76% of spending.** Cost per call follows context size, and context only grows.
- **Rewriting the cache after being away: 18%.** After an idle gap past the cache's lifetime, the next message rewrites the whole context, at 40 times the price of reading it on Opus 5.5. That happened only 26 times.

So what drives cost is how long sessions run, more than what's in them.

## What might work

These come from the same replay, using recorded token counts.

| Change | Effect on cost | Catch |
|---|---|---|
| Restart at the next prompt above 300K, bringing back 50K of context | -36% at most | Ignores rework, and any effect on the quality of the work |
| The same, bringing back 100K | -31% at most | The same |
| Refresh the cache through idle gaps for up to 8 hours | -3.0% at best | Needs something running the whole time |
| The same, for up to 48 hours | +4.6% | Most refreshes are wasted |

The restart figures are the first numbers near the original 30% target, but they're upper limits. A replay can't tell whether a fresh session would have done the work as well, or how much it would have had to re-learn.

So we tested it on a small scale: real Claude Code on 12 tasks, in one session or restarting every 4 tasks with a generated handoff. Nothing was lost, with every module passing and a rule from the first prompt still followed. But in small sessions, peaking around 55K tokens, it barely saved anything: 5% less on Haiku, and 16% *more* on Sonnet, because each restart rewrites its cache.

Then we made the sessions large by having each task read long design notes. With one restart on Sonnet, the session peaked at 271K tokens instead of 522K and cost $3.70 instead of $5.13, 28% less. Every module still passed, and a ticket number from every note reached the log. That's two runs per arm on synthetic tasks that don't depend on each other, which flatters a restart, so treat 28% as encouraging rather than settled. It's still the first real saving we measured that's near the original 30%, and it didn't come from compressing anything.

Keeping the cache warm is the idea we most expected to win. Refreshing is 40 times cheaper than rewriting, but people are away for a long time, and a session you never return to pays for every refresh.

## What we'd do differently

**Measure before building.** The replay would have shown most of this before any pruning rule existed. It's the most useful thing we made.

**Test live as well as offline.** Replay passed while a real task was failing. A pruning rule had dropped the only failure line, and only a live run found it. A cost ledger that replay never exercised also overstated cost by 56% until a live run caught it.

**Check what you're summing.** `claude -p` reports a running total when resuming. Adding those up made restarting look 60% cheaper in a pilot, when the real difference was 5%.

## Run it on your own sessions

Nothing leaves your machine, and neither command calls a model.

```sh
ccc whatif ~/.claude/projects
ccc bench ~/.claude/projects
```

If your numbers look different from these, we'd like to know. People who work in short sessions, or whose work is mostly reading logs, could well see something else.
