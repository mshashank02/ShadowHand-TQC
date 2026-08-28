# Repository structure

This repository separates reproducible inputs and source code from outputs created
by training, validation, and cluster jobs.

## Where files belong

- Put reusable GPU implementation in `shadowhand_gpu/` and mesh-processing code in
  `object_conversion/`.
- Keep stable command-line entrypoints at the repository root. This preserves the
  existing commands documented in `README.md` and used by cluster jobs.
- Put new Slurm job definitions in `jobs/slurm/`. Jobs should write scheduler output
  to `slurm_logs/` and runtime artifacts to `generated/` or an explicit artifact
  root.
- Keep source assets in `assets/`, `textures/`, or `stls/`. Keep study inputs and
  their manifests in `study_objects/`.
- Put CPU-independent tests in `tests/` and tests requiring the GPU stack in
  `tests_gpu/`.
- Put minimal issue reproductions in `reproducers/`, not in the production package.
- Treat `generated/`, `generated_objs/`, `logs/`, `models/`, `runs/`, `videos/`,
  `wandb/`, and new Slurm logs as local output directories.

## Tracked generated evidence

Some files already tracked below `generated/`, `logs/`, and `slurm_logs/archive/`
are historical validation evidence or fixtures consumed by tests. The ignore rules
prevent new outputs from being added accidentally; they do not remove existing
tracked evidence. Update such evidence deliberately with `git add -f <path>` and
explain the regeneration command in the commit or pull request.

## Naming conventions

- Python modules and scripts: `snake_case.py`.
- Tests: `test_<behavior>.py`.
- Slurm launchers: descriptive `snake_case.sbatch` names.
- Generated run directories: stable candidate/object/run identifiers rather than
  timestamps alone.

Run commands from the repository root unless an entrypoint explicitly says
otherwise. Slurm scripts determine the repository root before invoking Python, so
their location under `jobs/slurm/` does not change runtime paths.
