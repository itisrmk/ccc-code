# ccc-code plugin

Adds cost warnings, `/ccc-code:restart` and `/ccc-code:sharpen` to Claude Code, and local code search if the `ccc` program is installed.

```
/plugin marketplace add itisrmk/ccc-code
/plugin install ccc-code --marketplace itisrmk/ccc-code
```

The plugin only calls `ccc`, and it won't download anything by itself. If `ccc` isn't installed, it says so once in each session. See the [project README](https://github.com/itisrmk/ccc-code).
