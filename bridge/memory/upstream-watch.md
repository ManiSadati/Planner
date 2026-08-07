# Upstream Watch

Last updated: 2026-08-07

## Purpose

Track upstream and fork movement that can affect the NPU-IR-to-PTOAS bridge plan.

## Initial Exploration Ownership

The one-time initial exploration and one-month backfill are owned by Codex. They should not use the OpenAI API key.

## Daily Monitoring Ownership

The scheduled explorer agent will later run at 7:00am Eastern time, Monday through Saturday. It should use the OpenAI API key only for daily summarization of new changes since the last successful scan.

## PTOAS Tracking Targets

- `https://github.com/hw-native-sys/PTOAS`
- `https://github.com/zhendong404/PTOAS`
- `https://github.com/mouliangyu/PTOAS`
- `https://github.com/WenboCodes/PTOAS`
- forks of `https://github.com/mouliangyu/PTOAS`, especially active forks
- forks that are themselves forked from active forks, not only direct forks of upstream
- local fork: `/home/m84446336/PTOAS/PTOAS_Markham`

## PTOAS Source-Of-Truth Rule

The local PTOAS checkout is useful for building, running, and inspecting available code, but it is not authoritative for design state. Its `origin` remote is a personal fork, not upstream, and the checked-out branch may be far behind the active PTOAS ecosystem.

Explorer/backfill work must compare upstream, active forks, forks-of-forks, branches, issues, and PRs before deciding whether a PTOAS change matters for the bridge plan.

## NPU-IR Tracking Targets

- upstream source of truth: `https://gitcode.com/Ascend/AscendNPU-IR`
- main development fork: `https://gitcode.com/manisadati/AscendNPU-IR`
- local fork: `/home/m84446336/AscendNPU-IR`
- GitHub mirrors if found and useful

## Current Watch Status

- No one-month backfill report has been produced yet.
- No daily explorer report has been produced yet.
- No acknowledgement state exists yet because there has been no meaningful explorer report.
