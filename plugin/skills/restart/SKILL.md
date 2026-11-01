---
name: restart
description: Continue this work in a fresh session to cut what each message costs. Use when the person asks to restart or hand off a long session. Restarting only saved money past about 300K tokens of context, so say so if this session is smaller.
---

Run `ccc handoff --out .ccc-handoff.md` with Bash. Read the file and tell the person, in one or two lines, what it carries over and anything important it leaves out.

Then tell them to start the fresh session themselves, since a session can't replace itself:

```
ccc handoff --start
```

Delete `.ccc-handoff.md` afterward. If `ccc` isn't installed, say so and point to https://github.com/itisrmk/ccc-code.
