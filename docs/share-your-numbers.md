# Share your numbers

`ccc`'s results so far come from one developer, and two sessions make up most of the spending. Yours would show whether they hold for other people. This takes about five minutes, costs nothing and sends nothing anywhere. You read the files, then decide whether to send them.

## Run it

```sh
cargo install --git https://github.com/itisrmk/ccc-code ccc
ccc whatif --export whatif.json
ccc bench ~/.claude/projects --export bench.json
```

Neither command calls a model. `ccc bench` can take a few minutes on a large history.

## What the files hold

Totals only: call and session counts, shares of spending, and simulated costs. They hold no prompts, code, file paths, tool output or session ids, and a test checks that for each export. Open them and read them before sharing, because that's the check that counts.

The dollar figures are list prices. On a subscription they're notional.

## What would be useful to know alongside them

- Roughly how you work: long sessions or short ones, and how often you step away and come back
- What kind of work it is, such as a large application, a small library or data analysis
- Whether you use subagents much

## What the numbers can and can't show

`ccc whatif` replays recorded token counts, so it can say what a change would have cost. It can't say whether the work would have gone as well. Its restart figures are upper limits for that reason.
