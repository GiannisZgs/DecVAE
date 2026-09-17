# DecVAE — working agreement

This is an existing research repository under active revision on `v2-dev`, with new features being
added to code that already produces results. Treat it as a live codebase, not a greenfield project.

## Editing philosophy

- **Prefer careful, incremental edits.** Avoid drastic or sweeping changes that could destabilise
  the project. When a small, local change achieves the goal, use it instead of a refactor.
- **Avoid major changes where possible.** If a task seems to require a large restructuring, raise
  it first and agree on the scope before touching files.
- **Keep a backup when changing scripts** so a change can always be reverted (e.g. copy the file to
  `<name>.py.bak` or an equivalent alongside the original before editing it).
- **If unsure about anything, ask.** A question is cheaper than an unwanted change.

## Verifying changes

- Changing one part of the pipeline can affect parts that were not edited. After a change, check
  that the behaviour of the untouched code paths is the same as before — not only that the new
  behaviour works.
- Walk every branch a setting can take (model types, input types, argument values declared in
  `args_configs/`), not just the combination exercised by the example configs.

## Consistency with the existing project

- New files and edits must follow the structure and logic already established in the repository:
  the layout under `args_configs/`, `config_files/`, `models/`, `scripts/`, `data_preprocessing/`,
  `feature_extraction/`, and the naming and argument conventions used there.
- Match the surrounding code's style, idioms, and comment density rather than introducing a new one.
