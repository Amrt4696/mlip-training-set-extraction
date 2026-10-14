# AIMD to MLIP training set

Turns a VASP AIMD trajectory (OUTCAR) into a diverse, decorrelated
subset for training an MLIP (I use this for MACE).

## Why

Adjacent AIMD frames are nearly identical. Sampling every Nth frame
just pays DFT-level storage cost for near-duplicates. This picks a
subset actually spread out in configuration space, without losing the
rare structures (a high-force moment, an unusually compressed cell)
that a robust potential needs.

## What it does

- Stitches OUTCAR + any continuation OUTCARs into one trajectory, with
  provenance per frame.
- Auto-detects composition, atom count, species from the first frame;
  enforces it on every frame after.
- Windows to the later part of the trajectory (`--last`, `--stride`) -
  skips the still-equilibrating start.
- QC on every candidate: finite energy/forces/stress, max-force
  cutoff, min-distance cutoff. Rejections go to a CSV with the reason,
  nothing silent.
- SOAP descriptor per frame, averaged per species so no species
  dominates just by atom count.
- Final subset = physical-tail outliers (robust z-score on
  energy/atom, force, stress, volume) + even time coverage + SOAP
  farthest-point sampling, seeded by the first two.
- Writes extxyz + selection/rejected CSVs + summary, then reads the
  extxyz back and re-validates it before calling the run done.

## Usage

```
python3 scripts/extract_training_subset.py \
    --root /path/to/your/aimd/run \
    --system-label MgSiO3_postperovskite_160atom \
    --ensemble nvt \
    --pressure 100 --temperature 3000 \
    --output ./training_subset
```

Composition/atoms are auto-detected - `--system-label` is just a
free-text tag for filenames/metadata. `--pressure`/`--temperature` are
optional, recorded only if given.

NVT/NPT get different defaults for window size and selection counts
(`ENSEMBLE_DEFAULTS` in the script) - tuned for my own ~150-200 atom
runs, not universal. Override with
`--last/--stride/--number/--fps/--tail/--temporal`.

## What's here

- `scripts/extract_training_subset.py` - the extractor
- `notes/method.md` - why the selection is split into physical-tail /
  temporal / SOAP-FPS
- `notes/practical_notes.md` - OUTCAR/continuation gotchas
- `examples/MgSiO3_ppv_160atom_nvt_100GPa_2500K/` - a real 160-atom
  run, including a bug it caught and fixed in physical-tail selection

## Requirements

Python 3, `ase`, `dscribe`, `numpy`. No VASP-specific dependency beyond
ASE's OUTCAR reader, so this works on any AIMD trajectory ASE can read
that way - not just the systems I've actually run it on.

## Caveats

- Checks for numerically broken frames, not whether VASP's SCF actually
  converged at each ionic step. Check `NELM` vs. actual iteration
  counts yourself if you're not sure.
- SOAP defaults (`r_cut=5 A`, `n_max=6`, `l_max=4`, `sigma=0.5`) are
  tuned for oxide/silicate bond lengths, not universal.
- Physical-tail is a robust-statistics outlier score, not a guarantee
  of catching every interesting rare event.
