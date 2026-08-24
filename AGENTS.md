# Linex — working rules for AI assistants

## Goal of repo

This repo aims to refactor Molt (RL infra on NeMo Automodel Pytorch-native backend)
so all rollout related feature organically integrates with Polar (Harness native rollout
server that uses Slime as a reference trainer layer). 
The integration is double-sided. Molt uses only Polar as agent rollout backend and 
Polar uses Molt as the only training backend. All glue code and irrelevant code
(eg. Molt's own agent examples and Polar's dashboard and Slime & SGLang integrations) should be removed.
The only inference backend is vLLM router controlled by Molt's Ray. 
Duplicating arguments and data classes from both frameworks should merge into 
one and get configured in one place. CLI entry should still reuse Molt existing ones.

### Code Reference -- Molt and Polar

The original upstream of Molt (reference/labs-molt) and Polar (reference/ProRL-Agent-Server)
are cloned locally for read-only reference. Do not modify or treat them as dependency in 
any way. They are put here to demonstrate the original implementation 
of respective frameworks and to keep track of their latest changes.

## Code standards -- Simple and Elegant.

The priority order is explicit: # Simplicity is the first principle
**Human readability comes first; coding-agent traceability is the minimum
gate.** A human should understand the code in one pass. An agent must at least
be able to trace a feature from CLI flag to executed branch, tensor/record,
metric, and test without reconstructing hidden control flow. Every rule below
is an instance of that ordering. These are hard correctness rules, not style
preferences. A violation is a bug and must be fixed before the change ships.

1. **Code a human can't follow at a glance is a bug.** If a reviewer can't read
   a function top-to-bottom in one pass, restructure or delete it. Nesting,
   indirection, and clever constructs count against correctness — cleverness
   that costs comprehension is a defect, whatever it saves.

2. **Simplicity is the core engineering metric.** When two designs both work,
   ship the one with less code, fewer concepts, and fewer files. Prefer deleting
   code over adding it. A fix that adds more than ~20 lines for a problem
   statable in one sentence is suspect — find the smaller fix first. Never add
   config, record types, or return-shape changes "for the future".

3. **No over-encapsulation.** No new class / dataclass / helper / module for a
   single call site. A helper needs 3+ real call sites AND nontrivial logic —
   otherwise inline it. Never wrap trivial code. Never change a function
   signature or return shape to thread data that only one caller needs.

4. **Preserve intentional capabilities, not legacy API shapes.** Features,
   performance knobs, and observability are intentional — do not remove them
   in the name of simplicity. Knobs default ON stay ON; simplify their
   implementation without deleting their behavior.

5. **Framework elegance means one concept, one owner, and one obvious path.**
   Interfaces must follow domain boundaries, keep configuration and data
   ownership singular, and compose without exposing framework history or
   requiring glue code. Public surfaces must be minimal, orthogonal, and
   unsurprising.

6. **Backward compatibility does not outrank simplicity or elegance.** Do not
   preserve obsolete interfaces, adapters, or abstractions merely to avoid
   breaking changes. Rewrite them when a clean design requires it; retain
   compatibility only when a current, explicit requirement depends on it.

### Checklist before finishing any change

- Would a human reading this cold understand it in one pass? That is the gate.
- Does each concept have one owner and one obvious path through the system?
- Could this diff be half the size? If unsure, make it smaller.
- Any new class or file? Justify each with 3+ call sites, or delete it.
- Any signature / return-shape change? Verify every caller genuinely needs it.
- Comments: concise "why" only, 2-4 lines max, written for an external reader —
  no job ids, commit hashes, single-run metrics, or internal paths; keep
  upstream issue/PR links.
- One problem = one minimal diff. Do not batch unrelated "improvements".
- A "bug" that cannot trigger under the real recipes is not worth fixing.


## Agent Workflow (RFC --> Work Report)
If an agent is asked to implement a RFC under `rfcs/`, make sure to log important
challenges, caveat, experiment results, statistics, etc. When the implementation
completes, render a clear and instructive work report as webpage, containing
important data, visualization and include challenges and suggestions for the 
next steps. Put the report under `reports/rfc-{index}-{rfc_name}/`.  