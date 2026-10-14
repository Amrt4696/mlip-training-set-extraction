# Worked example: MgSiO3 post-perovskite, 160 atoms, NVT ~100 GPa / 2500 K

Real trajectory - 2000 steps, 1 fs timestep, VASP (`ISIF=2`,
`TEBEG=TEEND=2500`, fixed-cell NVT). OUTCAR (164 MiB) isn't committed -
too big, and it's exactly what this tool exists to shrink down. Lives
in `raw/` locally, gitignored.

Composition (Mg32 Si32 O96, 160 atoms) auto-detected, not told to the
script. Pressure/temperature are my own labels (external pressure in
the OUTCAR is ~1000 kB = ~100 GPa, `TEBEG=TEEND=2500` is right there)
- passed in with `--pressure`/`--temperature`, not computed.

```
python3 scripts/extract_training_subset.py \
    --root examples/MgSiO3_ppv_160atom_nvt_100GPa_2500K/raw \
    --system-label MgSiO3_postperovskite_160atom \
    --ensemble nvt \
    --pressure 100 --temperature 2500 \
    --output examples/MgSiO3_ppv_160atom_nvt_100GPa_2500K
```

## Result

2000 frames, one OUTCAR. Windowed to last 1000 at stride 2 -> 500
candidates, all passed QC. Final 70 = 50 SOAP-FPS + 10 physical-tail +
10 temporal, global frames 1004-1976. Energy/atom -6.228 to -6.135
eV/atom, max force 8.2-18.5 eV/A.

## The bug this run caught

First run: max SOAP cosine similarity among the 70 came back
0.9999998802 - basically a duplicate. Traced to two physical-tail
picks 2 fs apart (frames 1854/1856) - same force excursion, both hit
the top of the outlier ranking, and nothing checked whether tail picks
overlapped each other. Real gap in `choose_physical_tail`, fixed
there (see git history, `notes/method.md`) rather than patched here.

After the fix: 0.9999980115, ~10x further apart. Closest pair is still
two physical-tail picks (1336/1338, 4 fs apart) - now an honest
reflection of how clustered this trajectory's few outlier events are,
not a duplicate. If your trajectory has fewer independent excursions
than `--tail` asks for, it fails loudly instead of handing back
redundant structures - lower `--tail` or
`--tail-dup-relative-threshold`.

## Files here

- `raw/OUTCAR` - source trajectory (not committed)
- `*.extxyz` - 70 selected structures with REF_energy/forces/stress +
  provenance
- `*_selection.csv` - per-structure properties and selection method
- `*_rejected.csv` - empty, 0 rejected
- `*_summary.txt` - full run report
