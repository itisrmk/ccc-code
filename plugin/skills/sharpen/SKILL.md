---
name: sharpen
description: Turn a vague request into a specific one using this repository's code. Use when the person asks to sharpen or clarify a request before starting work.
argument-hint: <request>
---

The request to sharpen is: $ARGUMENTS

Run `ccc enhance "<the request>"` with Bash to see which files may be relevant. Check those files briefly if needed. Then write the request again so that it names the files, behaviour and how to check the result, keeping its meaning and adding no requirements.

Show the sharper version and ask whether to go ahead with it. Don't start the work until they say so.
