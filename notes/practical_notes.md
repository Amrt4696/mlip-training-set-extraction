# Practical notes on VASP OUTCARs and continuations

**Continuation directories.** Long AIMD runs rarely finish in one
SLURM allocation - walltime hits, VASP gets killed, you restart from
the last `CONTCAR`/`WAVECAR` in a fresh directory. Each restart has
its own OUTCAR covering only the steps it ran. `discover_segments()`
stitches them in order as long as they're named something it
recognizes (`cont_1`, `cont_2`, `continue_3`...) - natural-sorted, so
`cont_10` doesn't land before `cont_2`.

**Empty OUTCAR = skipped, not an error.** A killed-before-it-started
job or an unused restart directory just gets left out.

**Global frame index is into the combined trajectory**, not any one
OUTCAR. `source_segment`/`source_segment_frame` trace a selected
structure back to which raw OUTCAR it came from.

**Composition is checked on every candidate, not once** - catches a
restart that picked up the wrong CONTCAR or a species-ordering
mismatch between segments, logged to `rejected.csv` instead of
silently polluting the training set.

**Doesn't check SCF convergence per ionic step** - see the README
caveat. Catches NaNs/absurd forces, not "VASP finished the step but
didn't hit EDIFF."
