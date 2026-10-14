#!/usr/bin/env python3

from collections import namedtuple
from pathlib import Path
import argparse
import csv
import re
import sys

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP

# ============================================================
# AIMD TRAJECTORY -> DIVERSE, DECORRELATED MLIP TRAINING SUBSET
#
# Run from a directory containing one AIMD trajectory (an OUTCAR,
# plus any walltime-continuation OUTCARs in subdirectories).
#
# What it does, in one line: AIMD snapshots next to each other in
# time are nearly identical, so grabbing every Nth frame wastes
# training budget on near-duplicates. This instead windows down to
# the later (better-equilibrated) part of the trajectory, throws out
# anything that looks numerically broken, and then picks a subset
# that is deliberately diverse - some structural outliers, some
# frames just for even time coverage, and the rest chosen to be as
# different from each other as possible in SOAP-descriptor space.
#
# Composition, atom count and species are auto-detected from the
# first valid frame of the trajectory - this script isn't tied to
# one system. Everything downstream (SOAP, selection, output) is
# generic; you point it at whatever AIMD run you have.
#
# Candidate-window / selection-count defaults differ by ensemble
# (NVT vs NPT) because that's how much of each kind of trajectory
# I've found it actually takes to get a decorrelated sample - override
# any of them with --last/--stride/--number/--fps/--tail/--temporal
# if your trajectories are shorter/longer or you want more/fewer
# structures.
#
# Labels:
#   REF_energy : eV
#   REF_forces : eV/Angstrom
#   REF_stress : eV/Angstrom^3
#
# Stress Voigt order:
#   xx yy zz yz xz xy
# ============================================================

ENSEMBLE_DEFAULTS = {
    "nvt": {
        "number": 70,
        "last": 1000,
        "stride": 2,
        "fps": 50,
        "tail": 10,
        "temporal": 10,
    },
    "npt": {
        "number": 30,
        "last": 600,
        "stride": 2,
        "fps": 20,
        "tail": 5,
        "temporal": 5,
    },
}

# What the trajectory is expected to look like, detected once from
# the first valid frame and then enforced on every frame after that.
ExpectedStructure = namedtuple(
    "ExpectedStructure",
    ["natoms", "composition", "species_order"],
)


def die(msg):
    print()
    print("=" * 80, file=sys.stderr)
    print("ERROR", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    sys.exit(1)


def composition(atoms):
    result = {}
    for symbol in atoms.get_chemical_symbols():
        result[symbol] = result.get(symbol, 0) + 1
    return result


def detect_expected_structure(atoms):
    """Build the composition/species/atom-count fingerprint from the
    first valid frame of the trajectory. species_order preserves the
    order species first appear in (same convention VASP's POSCAR/
    CONTCAR groups atoms in), which is what the SOAP descriptor and
    the output filenames use."""
    symbols = atoms.get_chemical_symbols()

    species_order = list(dict.fromkeys(symbols))

    return ExpectedStructure(
        natoms=len(atoms),
        composition=composition(atoms),
        species_order=species_order,
    )


def check_composition(atoms, expected):
    got = composition(atoms)

    if len(atoms) != expected.natoms:
        return False, (
            f"atom count = {len(atoms)}, expected {expected.natoms}"
        )

    if got != expected.composition:
        return False, (
            f"composition = {got}, expected {expected.composition}"
        )

    return True, ""


def natural_key(path):
    name = path.name.lower()
    return [
        int(x) if x.isdigit() else x
        for x in re.split(r"(\d+)", name)
    ]


# ============================================================
# TRAJECTORY SEGMENTS
# ============================================================

def discover_segments(root):
    root = Path(root).resolve()

    found = []
    seen = set()

    main = root / "OUTCAR"
    if main.is_file() and main.stat().st_size > 0:
        found.append(("main", main))
        seen.add(main.resolve())

    continuation_dirs = []

    patterns = [
        "cont_*",
        "cont-*",
        "continue*",
        "continuation*",
    ]

    for pattern in patterns:
        for d in root.glob(pattern):
            if not d.is_dir():
                continue

            outcar = d / "OUTCAR"
            if not outcar.is_file() or outcar.stat().st_size == 0:
                continue

            real = outcar.resolve()
            if real in seen:
                continue

            seen.add(real)
            continuation_dirs.append(d)

    continuation_dirs = sorted(
        continuation_dirs,
        key=natural_key,
    )

    for d in continuation_dirs:
        found.append((d.name, d / "OUTCAR"))

    return found


def read_segments(segment_files):
    all_frames = []
    provenance = []
    global_index = 0

    print()
    print("=" * 80)
    print("TRAJECTORY SEGMENTS")
    print("=" * 80)

    for segment_number, (segment_name, path) in enumerate(
        segment_files,
        start=1,
    ):
        print()
        print(f"Segment      : {segment_number}")
        print(f"Name         : {segment_name}")
        print(f"OUTCAR       : {path}")
        print(f"Size         : {path.stat().st_size / 1024**2:.1f} MiB")

        try:
            frames = read(
                str(path),
                index=":",
                format="vasp-out",
            )
        except Exception as exc:
            die(
                f"ASE failed reading:\n{path}\n\n{exc}"
            )

        nseg = len(frames)
        print(f"Ionic frames : {nseg}")

        if nseg == 0:
            print("WARNING: zero ionic frames -> skipped")
            continue

        global_first = global_index
        global_last = global_index + nseg - 1

        print(
            f"Global range : {global_first} .. {global_last}"
        )

        for local_index, atoms in enumerate(frames):
            all_frames.append(atoms)
            provenance.append({
                "global_frame": global_index,
                "segment": segment_name,
                "segment_frame": local_index,
                "source_file": str(path),
            })
            global_index += 1

    if not all_frames:
        die("No ionic frames were found.")

    print()
    print("=" * 80)
    print(f"TOTAL COMBINED IONIC FRAMES = {len(all_frames)}")
    print("=" * 80)

    return all_frames, provenance


# ============================================================
# DFT LABELS / QC
# ============================================================

def get_labels(atoms, expected):
    if atoms.calc is None:
        raise RuntimeError(
            "ASE calculator / DFT labels are missing"
        )

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(
        atoms.get_stress(voigt=True),
        dtype=float,
    )

    if forces.shape != (expected.natoms, 3):
        raise RuntimeError(
            f"force shape = {forces.shape}; "
            f"expected ({expected.natoms},3)"
        )

    if stress.shape != (6,):
        raise RuntimeError(
            f"stress shape = {stress.shape}; expected (6,)"
        )

    return energy, forces, stress


def minimum_distance(atoms):
    try:
        i, j, d = neighbor_list(
            "ijd",
            atoms,
            cutoff=3.0,
        )
        mask = i != j
        if np.any(mask):
            return float(np.min(d[mask]))
    except Exception:
        pass

    dist = atoms.get_all_distances(mic=True)
    dist[dist < 1.0e-12] = np.inf
    return float(np.min(dist))


def validate_record(atoms, expected, max_force_cutoff, min_distance_cutoff):
    ok, reason = check_composition(atoms, expected)
    if not ok:
        return None, reason

    if not np.all(atoms.pbc):
        return None, "non-periodic structure"

    volume = float(atoms.get_volume())
    if not np.isfinite(volume) or volume <= 0.0:
        return None, f"invalid volume = {volume}"

    try:
        energy, forces, stress = get_labels(atoms, expected)
    except Exception as exc:
        return None, f"label error: {exc}"

    if not np.isfinite(energy):
        return None, "non-finite energy"

    if not np.all(np.isfinite(forces)):
        return None, "non-finite forces"

    if not np.all(np.isfinite(stress)):
        return None, "non-finite stress"

    force_norms = np.linalg.norm(forces, axis=1)
    max_force = float(np.max(force_norms))

    if max_force > max_force_cutoff:
        return None, (
            f"max force = {max_force:.6f} eV/A "
            f"> cutoff {max_force_cutoff:.6f}"
        )

    dmin = minimum_distance(atoms)

    if not np.isfinite(dmin) or dmin < min_distance_cutoff:
        return None, (
            f"minimum distance = {dmin:.6f} A "
            f"< cutoff {min_distance_cutoff:.6f}"
        )

    return {
        "atoms": atoms,
        "energy": energy,
        "forces": forces,
        "stress": stress,
        "volume": volume,
        "max_force": max_force,
        "min_distance": dmin,
    }, ""


# ============================================================
# CANDIDATE WINDOW
# ============================================================

def build_candidate_indices(nframes, last, stride):
    if last < 1:
        die("--last must be >= 1")

    if stride < 1:
        die("--stride must be >= 1")

    start = max(0, nframes - last)

    indices = list(
        range(start, nframes, stride)
    )

    if not indices:
        die("Candidate window is empty.")

    return indices


# ============================================================
# SOAP DESCRIPTORS
# ============================================================

def species_balanced_soap(records, expected, r_cut, n_max, l_max, sigma):
    soap = SOAP(
        species=expected.species_order,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        sparse=False,
        average="off",
    )

    descriptors = []

    print()
    print(
        f"Computing species-balanced SOAP "
        f"(r_cut={r_cut} A, n_max={n_max}, l_max={l_max}) ..."
    )

    for ii, record in enumerate(records, start=1):
        atoms = record["atoms"]

        local = np.asarray(
            soap.create(atoms),
            dtype=float,
        )

        symbols = np.asarray(
            atoms.get_chemical_symbols()
        )

        species_blocks = []

        for species in expected.species_order:
            mask = symbols == species

            if not np.any(mask):
                die(
                    f"SOAP error: species {species} absent"
                )

            block = np.mean(local[mask], axis=0)

            norm = np.linalg.norm(block)
            if norm > 0.0:
                block = block / norm

            species_blocks.append(block)

        descriptor = np.concatenate(species_blocks)

        norm = np.linalg.norm(descriptor)
        if norm > 0.0:
            descriptor = descriptor / norm

        descriptors.append(descriptor)

        if ii % 50 == 0 or ii == len(records):
            print(f"  SOAP {ii}/{len(records)}")

    return np.asarray(descriptors, dtype=float)


# ============================================================
# PHYSICAL-TAIL / TEMPORAL / FPS SELECTION
# ============================================================

def robust_scale(values):
    x = np.asarray(values, dtype=float)

    median = np.median(x)
    mad = np.median(np.abs(x - median))

    if mad > 1.0e-14:
        return (x - median) / (1.4826 * mad)

    std = np.std(x)
    if std > 1.0e-14:
        return (x - median) / std

    return np.zeros_like(x)


def choose_physical_tail(records, expected, number, descriptors, dup_relative_threshold=0.1):
    """Rank candidates by how much of a physical outlier they are, but
    skip any candidate that's a near-duplicate of one already picked.

    Without this, a single brief force/energy excursion can place
    several time-adjacent frames at the top of the ranking simultaneously
    - they're all "the outlier", because they're all the same outlier
    event a couple of femtoseconds apart. Picking several of them does
    nothing but burn slots that a genuinely different structure could
    have used (confirmed on a real trajectory: two tail-selected frames
    2 fs apart came out with SOAP cosine similarity 0.9999999 - effectively
    one structure counted twice).

    "Near-duplicate" is calibrated relative to this dataset's own
    pairwise-similarity scale, not a fixed absolute cosine value. For a
    large, mostly-rigid bulk crystal, species-averaged SOAP vectors for
    *any* two candidate frames can already sit above 0.9999 just from how
    compressed the descriptor space is at that size - checked on a real
    160-atom trajectory, where even frames 400 fs apart averaged 0.99998
    similarity. A fixed threshold like "similarity > 0.999" would flag
    nothing on that system, or everything on a smaller/noisier one. So
    instead this computes the median pairwise SOAP distance (1 -
    similarity) across all candidates, and treats two frames as
    duplicates only once their distance drops to a small fraction
    (`dup_relative_threshold`, default 10%) of that typical distance -
    i.e. they're candidates that are unusually close to each other even
    by this dataset's own already-compressed standards."""
    if number <= 0:
        return []

    energy_per_atom = np.asarray([
        rec["energy"] / expected.natoms
        for rec in records
    ])

    max_force = np.asarray([
        rec["max_force"]
        for rec in records
    ])

    stress_norm = np.asarray([
        np.linalg.norm(rec["stress"])
        for rec in records
    ])

    volume = np.asarray([
        rec["volume"]
        for rec in records
    ])

    score = (
        np.abs(robust_scale(energy_per_atom))
        + np.abs(robust_scale(max_force))
        + np.abs(robust_scale(stress_norm))
        + np.abs(robust_scale(volume))
    )

    order = np.argsort(score)[::-1]

    full_similarity = descriptors @ descriptors.T
    iu = np.triu_indices(len(descriptors), k=1)
    pairwise_distance = 1.0 - full_similarity[iu]
    median_distance = float(np.median(pairwise_distance))
    dup_distance_cutoff = dup_relative_threshold * median_distance

    chosen = []
    for idx in order:
        idx = int(idx)

        is_duplicate = any(
            (1.0 - float(descriptors[idx] @ descriptors[c])) < dup_distance_cutoff
            for c in chosen
        )
        if is_duplicate:
            continue

        chosen.append(idx)
        if len(chosen) == number:
            break

    if len(chosen) < number:
        die(
            f"Only found {len(chosen)} physical-tail outliers that aren't "
            f"near-duplicates of each other (needed {number}). The outlier "
            "events in this trajectory are too clustered - widen --last, "
            "increase --stride, lower --tail, or lower "
            "--tail-dup-relative-threshold."
        )

    return chosen


def choose_temporal(nrecords, number, forbidden=None):
    if number <= 0:
        return []

    forbidden = set() if forbidden is None else set(forbidden)

    targets = np.linspace(
        0,
        nrecords - 1,
        number + 2,
    )[1:-1]

    chosen = []
    all_indices = np.arange(nrecords, dtype=float)

    for target in targets:
        order = np.argsort(
            np.abs(all_indices - target)
        )

        found = False

        for idx in order:
            idx = int(idx)

            if idx in forbidden or idx in chosen:
                continue

            chosen.append(idx)
            found = True
            break

        if not found:
            die(
                "Could not find enough temporal-coverage structures."
            )

    return chosen


def fps(descriptors, target_total, seeds=None):
    n = len(descriptors)

    if target_total > n:
        die(
            f"Requested {target_total} selected structures "
            f"but only {n} valid candidates exist."
        )

    seeds = [] if seeds is None else seeds

    selected = list(
        dict.fromkeys(int(i) for i in seeds)
    )

    if len(selected) > target_total:
        die(
            "Number of reserved seed structures exceeds final total."
        )

    if not selected:
        selected = [0]

    similarity = descriptors @ descriptors.T

    distance_to_selected = np.min(
        1.0 - similarity[:, selected],
        axis=1,
    )

    for idx in selected:
        distance_to_selected[idx] = -np.inf

    while len(selected) < target_total:
        next_idx = int(np.argmax(distance_to_selected))

        if not np.isfinite(distance_to_selected[next_idx]):
            die("FPS failed: no finite candidate remains.")

        selected.append(next_idx)

        new_distance = 1.0 - similarity[:, next_idx]

        distance_to_selected = np.minimum(
            distance_to_selected,
            new_distance,
        )

        for idx in selected:
            distance_to_selected[idx] = -np.inf

    return selected


def max_cosine_similarity(descriptors):
    if len(descriptors) < 2:
        return np.nan

    matrix = descriptors @ descriptors.T
    np.fill_diagonal(matrix, -np.inf)

    return float(np.max(matrix))


# ============================================================
# OUTPUT
# ============================================================

def attach_output_labels(
    record,
    system_label,
    pressure,
    temperature,
    ensemble,
    source_meta,
    selection_method,
):
    atoms = record["atoms"].copy()
    atoms.calc = None

    atoms.info["REF_energy"] = float(record["energy"])

    atoms.arrays["REF_forces"] = np.asarray(
        record["forces"],
        dtype=float,
    )

    atoms.info["REF_stress"] = np.asarray(
        record["stress"],
        dtype=float,
    )

    config_type = f"{system_label}_{ensemble.upper()}"
    if pressure is not None:
        config_type += f"_{pressure}GPa"
    if temperature is not None:
        config_type += f"_{temperature}K"
    atoms.info["config_type"] = config_type

    atoms.info["source_system"] = system_label
    atoms.info["source_family"] = "aimd"
    atoms.info["source_ensemble"] = ensemble.upper()

    if pressure is not None:
        atoms.info["source_pressure_GPa"] = float(pressure)
    if temperature is not None:
        atoms.info["source_temperature_K"] = int(temperature)

    atoms.info["source_global_frame"] = int(
        source_meta["global_frame"]
    )

    atoms.info["source_segment"] = str(
        source_meta["segment"]
    )

    atoms.info["source_segment_frame"] = int(
        source_meta["segment_frame"]
    )

    atoms.info["selection_method"] = selection_method
    atoms.info["source_file"] = str(source_meta["source_file"])

    return atoms


def verify_extxyz(path, expected, expected_count, pressure, temperature, ensemble):
    try:
        frames = read(
            str(path),
            index=":",
            format="extxyz",
        )
    except Exception as exc:
        die(
            f"Cannot read back extxyz:\n{path}\n\n{exc}"
        )

    if len(frames) != expected_count:
        die(
            f"Read-back frame count = {len(frames)}, "
            f"expected {expected_count}"
        )

    required_keys = [
        "REF_energy",
        "REF_stress",
        "source_ensemble",
        "selection_method",
    ]
    if pressure is not None:
        required_keys.append("source_pressure_GPa")
    if temperature is not None:
        required_keys.append("source_temperature_K")

    for i, atoms in enumerate(frames):
        ok, reason = check_composition(atoms, expected)
        if not ok:
            die(f"Read-back frame {i}: {reason}")

        for key in required_keys:
            if key not in atoms.info:
                die(
                    f"Read-back frame {i}: {key} missing"
                )

        if "REF_forces" not in atoms.arrays:
            die(
                f"Read-back frame {i}: REF_forces missing"
            )

        energy = np.asarray(atoms.info["REF_energy"])
        stress = np.asarray(atoms.info["REF_stress"])
        forces = np.asarray(atoms.arrays["REF_forces"])

        if not np.all(np.isfinite(energy)):
            die(f"Read-back frame {i}: non-finite energy")

        if not np.all(np.isfinite(forces)):
            die(f"Read-back frame {i}: non-finite forces")

        if not np.all(np.isfinite(stress)):
            die(f"Read-back frame {i}: non-finite stress")

        if forces.shape != (expected.natoms, 3):
            die(
                f"Read-back frame {i}: force shape = {forces.shape}"
            )

        if stress.shape != (6,):
            die(
                f"Read-back frame {i}: stress shape = {stress.shape}"
            )

        ens = str(atoms.info["source_ensemble"]).upper()
        if ens != ensemble.upper():
            die(
                f"Read-back frame {i}: ensemble metadata {ens} "
                f"!= {ensemble.upper()}"
            )

        if pressure is not None:
            p = float(atoms.info["source_pressure_GPa"])
            if abs(p - pressure) > 1.0e-8:
                die(
                    f"Read-back frame {i}: pressure metadata {p} != {pressure}"
                )

        if temperature is not None:
            t = int(atoms.info["source_temperature_K"])
            if t != temperature:
                die(
                    f"Read-back frame {i}: temperature metadata {t} != {temperature}"
                )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pick a diverse, decorrelated subset of AIMD configurations "
            "from a VASP OUTCAR trajectory for MLIP training."
        )
    )

    parser.add_argument(
        "--system-label",
        required=True,
        help=(
            "free-text tag for this system, used in output filenames "
            "and metadata (e.g. MgSiO3_postperovskite_160atom)"
        ),
    )

    parser.add_argument(
        "--ensemble",
        required=True,
        choices=["nvt", "npt", "NVT", "NPT"],
    )

    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help="GPa, optional - recorded in metadata/filename if given",
    )

    parser.add_argument(
        "--temperature",
        type=int,
        default=None,
        help="K, optional - recorded in metadata/filename if given",
    )

    parser.add_argument(
        "--root",
        default=".",
        help="trajectory directory; default = current directory",
    )

    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help=(
            "number of final combined ionic frames to consider; "
            "default depends on ensemble"
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="candidate stride; default depends on ensemble",
    )

    parser.add_argument(
        "--number",
        type=int,
        default=None,
        help="number selected; default NVT=70, NPT=30",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--tail",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--temporal",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-force",
        type=float,
        default=50.0,
        help=(
            "hard pathological-force cutoff in eV/A; default = 50"
        ),
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.0,
        help=(
            "minimum allowed interatomic distance in A, catches bad "
            "geometries; default = 1.0 (loosen for systems with "
            "genuinely short bonds, e.g. hydrides)"
        ),
    )

    parser.add_argument(
        "--soap-r-cut",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--soap-n-max",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--soap-l-max",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--soap-sigma",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--tail-dup-relative-threshold",
        type=float,
        default=0.1,
        help=(
            "two physical-tail candidates are treated as near-duplicates "
            "(only one kept) once their SOAP distance drops below this "
            "fraction of the dataset's own median pairwise distance; "
            "default = 0.1. Relative, not absolute, because how "
            "'compressed' SOAP similarity is near 1.0 depends heavily on "
            "system size - see choose_physical_tail() docstring"
        ),
    )

    parser.add_argument(
        "--output",
        default="./training_subset",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    ensemble = args.ensemble.lower()
    defaults = ENSEMBLE_DEFAULTS[ensemble]

    number = defaults["number"] if args.number is None else args.number
    last = defaults["last"] if args.last is None else args.last
    stride = defaults["stride"] if args.stride is None else args.stride
    fps_n = defaults["fps"] if args.fps is None else args.fps
    tail_n = defaults["tail"] if args.tail is None else args.tail
    temporal_n = defaults["temporal"] if args.temporal is None else args.temporal

    if fps_n + tail_n + temporal_n != number:
        die(
            "--fps + --tail + --temporal must equal --number\n"
            f"Got {fps_n} + {tail_n} + {temporal_n} != {number}"
        )

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    if not root.is_dir():
        die(
            f"Trajectory root does not exist:\n{root}"
        )

    output.mkdir(parents=True, exist_ok=True)

    base = f"{args.system_label}_{ensemble}_n{number}"
    if args.pressure is not None:
        base += f"_{args.pressure:g}GPa"
    if args.temperature is not None:
        base += f"_{args.temperature}K"

    xyz_file = output / f"{base}.extxyz"
    selection_file = output / f"{base}_selection.csv"
    rejected_file = output / f"{base}_rejected.csv"
    summary_file = output / f"{base}_summary.txt"

    output_files = [
        xyz_file,
        selection_file,
        rejected_file,
        summary_file,
    ]

    if not args.overwrite:
        existing = [p for p in output_files if p.exists()]
        if existing:
            die(
                "Output already exists.\n"
                "Use --overwrite only if replacement is intentional.\n\n"
                + "\n".join(str(p) for p in existing)
            )

    print()
    print("=" * 80)
    print("AIMD TRAINING SUBSET EXTRACTOR")
    print("=" * 80)
    print(f"system label   : {args.system_label}")
    print(f"ensemble       : {ensemble.upper()}")
    if args.pressure is not None:
        print(f"pressure       : {args.pressure} GPa")
    if args.temperature is not None:
        print(f"temperature    : {args.temperature} K")
    print(f"root           : {root}")
    print(f"last frames    : {last}")
    print(f"stride         : {stride}")
    print(f"target         : {number}")
    print(
        f"selection      : {fps_n} SOAP-FPS + "
        f"{tail_n} physical-tail + {temporal_n} temporal"
    )
    print(f"max-force QC   : {args.max_force} eV/A")
    print(f"min-distance QC: {args.min_distance} A")
    print(f"output         : {output}")
    print("=" * 80)

    segments = discover_segments(root)

    if not segments:
        die(
            f"No usable OUTCAR found in:\n{root}\n"
            "or continuation directories."
        )

    print()
    print("Discovered segment order:")

    for i, (name, path) in enumerate(segments, start=1):
        print(f"  {i:2d}. {name:20s} {path}")

    frames, provenance = read_segments(segments)

    expected = detect_expected_structure(frames[0])

    print()
    print("=" * 80)
    print("DETECTED STRUCTURE (from first frame)")
    print("=" * 80)
    print(f"atoms       : {expected.natoms}")
    print(f"composition : {expected.composition}")
    print(f"species     : {expected.species_order}")

    # First/last raw composition sanity check against what frame 0 set.
    for label, idx in [
        ("first", 0),
        ("last", len(frames) - 1),
    ]:
        ok, reason = check_composition(frames[idx], expected)
        if not ok:
            die(
                f"{label} raw frame failed composition check:\n{reason}"
            )

    candidate_indices = build_candidate_indices(
        len(frames),
        last,
        stride,
    )

    print()
    print("=" * 80)
    print("CANDIDATE WINDOW")
    print("=" * 80)
    print(f"Combined raw frames    : {len(frames)}")
    print(
        f"Candidate global range : "
        f"{candidate_indices[0]} .. {candidate_indices[-1]}"
    )
    print(f"Candidate stride       : {stride}")
    print(f"Candidates before QC   : {len(candidate_indices)}")

    valid_records = []
    valid_meta = []
    rejected = []

    for global_idx in candidate_indices:
        record, reason = validate_record(
            frames[global_idx],
            expected,
            args.max_force,
            args.min_distance,
        )

        meta = provenance[global_idx]

        if record is None:
            rejected.append({
                "global_frame": global_idx,
                "segment": meta["segment"],
                "segment_frame": meta["segment_frame"],
                "reason": reason,
                "source_file": meta["source_file"],
            })
            continue

        record["global_frame"] = global_idx
        valid_records.append(record)
        valid_meta.append(meta)

    print(f"Valid candidates       : {len(valid_records)}")
    print(f"Rejected candidates    : {len(rejected)}")

    if len(valid_records) < number:
        die(
            f"Only {len(valid_records)} valid candidates remain; "
            f"{number} are required.\n"
            "Use a larger --last window and/or smaller --stride."
        )

    descriptors = species_balanced_soap(
        valid_records,
        expected,
        args.soap_r_cut,
        args.soap_n_max,
        args.soap_l_max,
        args.soap_sigma,
    )

    tail_indices = choose_physical_tail(
        valid_records,
        expected,
        tail_n,
        descriptors,
        dup_relative_threshold=args.tail_dup_relative_threshold,
    )

    temporal_indices = choose_temporal(
        len(valid_records),
        temporal_n,
        forbidden=tail_indices,
    )

    seeds = list(
        dict.fromkeys(tail_indices + temporal_indices)
    )

    final_indices = fps(
        descriptors,
        number,
        seeds=seeds,
    )

    if len(final_indices) != number:
        die("Final selection count mismatch.")

    methods = {}

    for idx in tail_indices:
        methods[idx] = "physical_tail"

    for idx in temporal_indices:
        methods[idx] = "temporal_coverage"

    for idx in final_indices:
        if idx not in methods:
            methods[idx] = "soap_fps"

    n_tail = sum(
        methods[i] == "physical_tail"
        for i in final_indices
    )

    n_temporal = sum(
        methods[i] == "temporal_coverage"
        for i in final_indices
    )

    n_fps = sum(
        methods[i] == "soap_fps"
        for i in final_indices
    )

    selected_descriptors = descriptors[final_indices]
    max_similarity = max_cosine_similarity(selected_descriptors)

    selected_atoms = []
    selection_rows = []

    for idx in final_indices:
        record = valid_records[idx]
        meta = valid_meta[idx]
        method = methods[idx]

        atoms_out = attach_output_labels(
            record=record,
            system_label=args.system_label,
            pressure=args.pressure,
            temperature=args.temperature,
            ensemble=ensemble,
            source_meta=meta,
            selection_method=method,
        )

        selected_atoms.append(atoms_out)

        stress = record["stress"]

        row = {
            "system_label": args.system_label,
            "ensemble": ensemble.upper(),
            "global_frame": meta["global_frame"],
            "segment": meta["segment"],
            "segment_frame": meta["segment_frame"],
            "selection_method": method,
            "energy_eV": record["energy"],
            "energy_per_atom_eV": (
                record["energy"] / expected.natoms
            ),
            "max_force_eVA": record["max_force"],
            "volume_A3": record["volume"],
            "min_distance_A": record["min_distance"],
            "stress_xx_eVA3": stress[0],
            "stress_yy_eVA3": stress[1],
            "stress_zz_eVA3": stress[2],
            "stress_yz_eVA3": stress[3],
            "stress_xz_eVA3": stress[4],
            "stress_xy_eVA3": stress[5],
            "source_file": meta["source_file"],
        }

        if args.pressure is not None:
            row["pressure_GPa"] = args.pressure
        if args.temperature is not None:
            row["temperature_K"] = args.temperature

        selection_rows.append(row)

    write(
        str(xyz_file),
        selected_atoms,
        format="extxyz",
    )

    with selection_file.open("w", newline="") as f:
        fieldnames = list(selection_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_rows)

    with rejected_file.open("w", newline="") as f:
        fieldnames = [
            "global_frame",
            "segment",
            "segment_frame",
            "reason",
            "source_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rejected:
            writer.writerows(rejected)

    verify_extxyz(
        path=xyz_file,
        expected=expected,
        expected_count=number,
        pressure=args.pressure,
        temperature=args.temperature,
        ensemble=ensemble,
    )

    selected_global = [
        valid_meta[i]["global_frame"]
        for i in final_indices
    ]

    segment_lines = []
    for i, (name, path) in enumerate(segments, start=1):
        segment_lines.append(f"  {i}. {name}: {path}")

    segment_text = "\n".join(segment_lines)

    pressure_line = (
        f"pressure                    : {args.pressure} GPa\n"
        if args.pressure is not None else ""
    )
    temperature_line = (
        f"temperature                 : {args.temperature} K\n"
        if args.temperature is not None else ""
    )

    summary = f"""================================================================================
AIMD TRAINING SUBSET SELECTION SUMMARY
================================================================================

system label                : {args.system_label}
ensemble                    : {ensemble.upper()}
{pressure_line}{temperature_line}root                        : {root}
detected composition        : {expected.composition}
atoms/frame                 : {expected.natoms}

trajectory segments:
{segment_text}

number of OUTCAR segments   : {len(segments)}
raw ionic frames combined   : {len(frames)}
requested final window      : {last}
candidate first/last        : {candidate_indices[0]} / {candidate_indices[-1]}
candidate stride            : {stride}
candidate count before QC   : {len(candidate_indices)}
valid candidates            : {len(valid_records)}
rejected candidates         : {len(rejected)}
SOAP-FPS                    : {n_fps}
physical tails              : {n_tail}
temporal coverage           : {n_temporal}
TOTAL SELECTED              : {len(final_indices)}
selected global range       : {min(selected_global)} .. {max(selected_global)}
max SOAP cosine similarity  : {max_similarity:.10f}

Labels:
  REF_energy : eV
  REF_forces : eV/Angstrom
  REF_stress : eV/Angstrom^3

Stress Voigt order:
  xx yy zz yz xz xy

Metadata:
  source_system
  source_family
  source_ensemble
  source_pressure_GPa (if --pressure given)
  source_temperature_K (if --temperature given)
  source_global_frame
  source_segment
  source_segment_frame
  selection_method
  source_file

extxyz:
  {xyz_file}

selection CSV:
  {selection_file}

rejected CSV:
  {rejected_file}

extxyz read-back verification: PASS
STATUS = PASS
"""

    summary_file.write_text(summary)

    print()
    print("=" * 80)
    print("SELECTION COMPLETE")
    print("=" * 80)
    print(f"Ensemble       : {ensemble.upper()}")
    print(f"Selected       : {len(final_indices)}")
    print(f"SOAP-FPS       : {n_fps}")
    print(f"Physical-tail  : {n_tail}")
    print(f"Temporal       : {n_temporal}")
    print(
        f"Global range   : "
        f"{min(selected_global)} .. {max(selected_global)}"
    )
    print(
        f"Max SOAP cosine similarity: {max_similarity:.10f}"
    )
    print()
    print(f"Wrote: {xyz_file}")
    print(f"Wrote: {selection_file}")
    print(f"Wrote: {rejected_file}")
    print(f"Wrote: {summary_file}")
    print()
    print("extxyz read-back verification: PASS")
    print("STATUS = PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
