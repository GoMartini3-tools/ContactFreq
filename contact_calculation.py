#!/usr/bin/env python3
# updated: 24-09-2025
"""
Comprehensive contact analysis pipeline including martinize2.

Now supports frames in PDB or CIF natively, without conversion.
If CIF frames are present, they are used directly so chain IDs are preserved.

This script performs the following steps:
  1. Generate contact maps for each frame (.pdb or .cif)
  2. Clean and filter contacts by distance and flags
  3. Annotate intra and inter chain contacts
  4. Compute contact frequencies and identify high-frequency pairs
  5. Select the single reference frame with the most high-frequency contacts
  6. Run martinize2 to build coarse-grained topology and structure
  7. Build bead index, write mock ITP and filter real ITP
  8. Measure distances for missing contacts and write them to a separate ITP
  9. Write per-frame counts of high-frequency contacts and Go contacts
 10. Move final .txt, .map and frame files into an output_files folder

Usage:
  python contact_analysis.py [options]
  e.g. python contact_calculation.py --type both --merge all --dssp mkdssp --go-eps 15 --from charmm --cm /home/phoenix/software/
"""

import os
import glob
import shutil
import re
import argparse
import subprocess
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from datetime import datetime
from typing import Dict, List, Tuple

# ---------------- frame discovery ----------------

FRAME_RE = re.compile(r"^frame_(\d+)\.(pdb|cif)$", re.IGNORECASE)

def list_frames() -> Dict[int, str]:
    """Return {frame_index: path} for frame_####.(pdb|cif), preferring PDB if both exist for same index."""
    candidates: Dict[int, Tuple[str, str]] = {}
    for fn in glob.glob("frame_*.*"):
        m = FRAME_RE.match(os.path.basename(fn))
        if not m:
            continue
        idx, ext = int(m.group(1)), m.group(2).lower()
        if idx not in candidates:
            candidates[idx] = (fn, ext)
        else:
            # prefer pdb over cif when both exist
            if candidates[idx][1] == "cif" and ext == "pdb":
                candidates[idx] = (fn, ext)
    return {k: candidates[k][0] for k in sorted(candidates)}

# ---------------- minimal mmCIF reader (no MDAnalysis dependency for CIF) ----------------

def read_cif_atoms(path: str) -> List[Dict[str, str]]:
    """
    Minimal mmCIF atom_site reader without using file tell/seek.
    Returns list of dicts with keys: chain, resid, name, x, y, z
    Prefers auth_* fields; falls back to label_*.
    Assumes no whitespace-containing values for needed columns.
    """
    with open(path, "r") as fh:
        # strip blanks and comments for simpler parsing
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]

    rows: List[Dict[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        s = lines[i]
        if s != "loop_":
            i += 1
            continue

        # collect headers
        i += 1
        headers: List[str] = []
        while i < n and lines[i].startswith("_"):
            headers.append(lines[i])
            i += 1

        # need atom_site with required columns
        if not headers or not any(h.startswith("_atom_site.") for h in headers):
            # skip data rows of this loop
            while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
                i += 1
            continue

        idx = {h: k for k, h in enumerate(headers)}
        def pick(name_list):
            for nm in name_list:
                if nm in idx:
                    return idx[nm]
            return None

        i_chain = pick(["_atom_site.auth_asym_id", "_atom_site.label_asym_id"])
        i_resid = pick(["_atom_site.auth_seq_id", "_atom_site.label_seq_id"])
        i_name  = pick(["_atom_site.auth_atom_id", "_atom_site.label_atom_id"])
        i_x = idx.get("_atom_site.Cartn_x")
        i_y = idx.get("_atom_site.Cartn_y")
        i_z = idx.get("_atom_site.Cartn_z")

        if None in (i_chain, i_resid, i_name, i_x, i_y, i_z):
            # not the atom_site loop we need; skip its data rows
            while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
                i += 1
            continue

        # consume data rows for this loop
        while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
            tokens = lines[i].split()
            # only accept rows matching header length (simple tokenization)
            if len(tokens) >= len(headers):
                try:
                    chain = tokens[i_chain]
                    resid = tokens[i_resid]
                    name  = tokens[i_name]
                    x = float(tokens[i_x]); y = float(tokens[i_y]); z = float(tokens[i_z])
                    rows.append({"chain": chain, "resid": resid, "name": name, "x": x, "y": y, "z": z})
                except Exception:
                    pass
            i += 1

    return rows


def get_cif_chains(path: str) -> List[str]:
    atoms = read_cif_atoms(path)
    return sorted({a["chain"] for a in atoms if a["chain"]})

def get_cif_ca_coords(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Return {(resid_str, chain): np.array([x,y,z])} for CA atoms from a CIF frame (angstroms).
    """
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for a in read_cif_atoms(path):
        if a["name"].upper() == "CA":
            try:
                resid_str = str(int(float(a["resid"])))
            except Exception:
                resid_str = a["resid"]
            out[(resid_str, a["chain"])] = np.array([a["x"], a["y"], a["z"]], dtype=float)
    return out

# ---------------- core steps ----------------

def process_contact_map(args):
    in_file, cm_dir = args
    exe = os.path.join(cm_dir, "contact_map")
    base, _ = os.path.splitext(in_file)
    out_map = f"{base}.map"
    subprocess.run([exe, in_file],
                   stdout=open(out_map, "w"),
                   stderr=subprocess.DEVNULL)

def run_contact_map(frames, cm_dir, cpus):
    with Pool(cpus) as pool:
        for _ in tqdm(pool.imap_unordered(
                        process_contact_map,
                        [(p, cm_dir) for p in frames]
                    ),
                      total=len(frames),
                      desc="Mapping"):
            pass

def clean_maps(src, backup, header_regex):
    """
    Keep rows after the header line using a regex so spacing differences do not break parsing.
    """
    os.makedirs(backup, exist_ok=True)
    hdr_re = re.compile(header_regex)
    for m in glob.glob(os.path.join(src, "*.map")):
        bkp = os.path.join(backup, os.path.basename(m))
        shutil.move(m, bkp)
        with open(bkp) as inp, open(m, "w") as out:
            hit_header = False
            for line in inp:
                if not hit_header:
                    if hdr_re.search(line):
                        hit_header = True
                        out.write(line)
                else:
                    if "UNMAPPED" not in line:
                        out.write(line)

def filter_map(map_file, short, long, out_txt):
    ov = re.compile(r"1 [01] [01] [01]")
    rz = re.compile(r"[01] [01] [01] 1")
    with open(map_file) as f, open(out_txt, "w") as out:
        for line in f:
            if not line.startswith("R"):
                continue
            parts = line.split()
            try:
                i1, i2 = int(parts[5]), int(parts[9])       # I(PDB)
                dist = float(parts[10])
                flags = " ".join(parts[11:15])
                r1, c1 = parts[3], parts[4]                 # resname, chain
                r2, c2 = parts[7], parts[8]
            except (IndexError, ValueError):
                continue
            if (abs(i2 - i1) >= 4 and
                short <= dist <= long and
                (ov.search(flags) or rz.search(flags))):
                out.write(f"{r1}\t{c1}\t{i1}\t{r2}\t{c2}\t{i2}\t{dist:.4f}\t{flags}\n")

def annotate(inp, outp, keep_same, keep_diff):
    seen = set()
    with open(inp) as fin, open(outp, "w") as out:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) < 7:
                continue
            ch1, ch2 = cols[1], cols[4]
            if ((keep_same and ch1 == ch2) or
                (keep_diff and ch1 != ch2)):
                pair = (ch1, cols[2], ch2, cols[5])  # (c1, i1, c2, i2)
                inv = (ch2, cols[5], ch1, cols[2])
                if pair not in seen and inv not in seen:
                    seen.add(pair)
                    rel = ("same_chain" if ch1 == ch2 else "different_chains")
                    out.write(line.strip() + f"\t{rel}\n")

def analyze_frequency(pattern, out_norm, out_high, thr):
    """
    Build per-pair frequency over all annotated_* files.

    The normalized/high files have columns:
      Res1  Res2  Freq  Chain1  Chain2  Resname1  Resname2
    where Res1 and Res2 are i1 and i2 from the annotated lines,
    Chain1 and Chain2 are c1 and c2, and Resname1/2 are r1 and r2.
    """
    counts = defaultdict(int)
    records = []
    for fn in glob.glob(pattern):
        lines = open(fn).read().splitlines()
        records.append([l for l in lines if l and not l.startswith("Res1")])
    total = len(records) if records else 1
    for rec in records:
        for l in rec:
            c = l.split()
            if len(c) < 6:
                continue
            key = (c[2], c[5], c[1], c[4], c[0], c[3])  # (i1, i2, c1, c2, r1, r2)
            counts[key] += 1

    with open(out_norm, "w") as out:
        out.write("Res1\tRes2\tFreq\tChain1\tChain2\tResname1\tResname2\n")
        for k, v in counts.items():
            i1, i2, c1, c2, r1, r2 = k
            out.write(f"{i1}\t{i2}\t{v/total:.2f}\t{c1}\t{c2}\t{r1}\t{r2}\n")

    with open(out_high, "w") as out:
        header = open(out_norm).readline()
        out.write(header)
        for l in open(out_norm).read().splitlines()[1:]:
            if float(l.split()[2]) >= thr:
                out.write(l + "\n")

# ---------------- helpers for keys and per-frame counting ----------------

def _key_from_annotated_line(line):
    """
    From annotated_* line with columns:
      0:rname1 1:c1 2:i1_resid 3:rname2 4:c2 5:i2_resid ...
    Return orientation independent key (i_resid1, c1, i_resid2, c2).
    """
    p = line.split()
    if len(p) < 6:
        return None
    a = (p[2], p[1], p[5], p[4])
    b = (p[5], p[4], p[2], p[1])
    return a if a <= b else b

def write_counts_per_frame(ref_pairs, annotated_pattern, out_path, label="RefSet"):
    total_ref = len(ref_pairs) if ref_pairs else 1
    files = []
    for fn in glob.glob(annotated_pattern):
        m = re.search(r"frame_(\d+)", os.path.basename(fn))
        if m:
            files.append((int(m.group(1)), fn))
    files.sort(key=lambda x: x[0])

    with open(out_path, "w") as out:
        out.write(f"Frame\t{label}\tFractionOfRefSet\tFile\n")
        for idx, fn in files:
            cnt = 0
            with open(fn) as f:
                for L in f:
                    if not L.strip() or L.startswith("Res1"):
                        continue
                    k = _key_from_annotated_line(L)
                    if k and k in ref_pairs:
                        cnt += 1
            out.write(f"{idx}\t{cnt}\t{(cnt/total_ref):.4f}\t{os.path.basename(fn)}\n")

def go_pairs_as_resid_chain(itp_path, inv_rev):
    ref = set()
    for line in open(itp_path):
        if not line.startswith("molecule_0_"):
            continue
        a, b = line.split()[:2]
        i1 = int(a.rsplit("_", 1)[1])
        i2 = int(b.rsplit("_", 1)[1])
        if i1 in inv_rev and i2 in inv_rev:
            (res1, ch1) = inv_rev[i1]
            (res2, ch2) = inv_rev[i2]
            t1 = (res1, ch1, res2, ch2)
            t2 = (res2, ch2, res1, ch1)
            ref.add(t1 if t1 <= t2 else t2)
    return ref

def _key_from_high_line(line):
    """
    From high_* line with columns:
      0:i1 1:i2 2:freq 3:c1 4:c2 5:r1 6:r2
    Build an orientation-independent tuple (i1_resid, c1, i2_resid, c2).
    """
    p = line.split()
    if len(p) < 5:
        return None
    a = (p[0], p[3], p[1], p[4])
    b = (p[1], p[4], p[0], p[3])
    return a if a <= b else b

def write_high_counts_per_frame(highfile, annotated_pattern, out_path):
    high_keys = set()
    with open(highfile) as fh:
        next(fh, None)
        for line in fh:
            k = _key_from_high_line(line)
            if k:
                high_keys.add(k)
    write_counts_per_frame(high_keys, annotated_pattern, out_path, label="HighContacts")

# ---------------- martinize2 runner ----------------

def run_martinize_from_atom(atom_path,
                            go_map_path,
                            merge,
                            dssp,
                            goeps,
                            src,
                            posres,
                            ss,
                            nter_list,
                            cter_list,
                            neutral_termini):
    atom = atom_path
    base = os.path.splitext(os.path.basename(atom))[0]
    atom_dir = os.path.dirname(atom) or "."
    cg = os.path.join(atom_dir, f"{base}_CG.pdb")

    cmd = ["martinize2", "-f", atom]

    if merge:
        cmd += ["-merge", merge]
    if dssp:
        cmd += ["-dssp", dssp]
    if ss:
        cmd += ["-ss", ss]

    cmd += ["-go", go_map_path, "-go-eps", str(goeps)]

    for mod in (nter_list or []):
        cmd += ["-nter", mod]
    for mod in (cter_list or []):
        cmd += ["-cter", mod]
    if neutral_termini:
        cmd += ["-nt"]

    cmd += [
        "-o", "topol.top",
        "-x", cg,
        "-p", posres,
        "-ff", "martini3001",
        "-cys", "auto",
        "-ignh",
        "-from", src,
        "-name", "molecule_0",
        "-maxwarn", "100"
    ]

    print("Running martinize2:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return atom

# ---------------- index builder for PDB and CIF ----------------

def build_index(struct_path: str):
    """
    Map (resid_str, chain_id) -> sequential bead index.
    PDB is parsed by text. CIF is parsed with read_cif_atoms (no MDAnalysis).
    """
    ext = os.path.splitext(struct_path)[1].lower()
    inv, offset = {}, 0

    if ext == ".pdb":
        by_chain = defaultdict(list)
        with open(struct_path) as fh:
            for l in fh:
                if l.startswith(("ATOM", "HETATM")):
                    ch = l[21]
                    try:
                        resi = int(l[22:26])
                    except ValueError:
                        continue
                    by_chain[ch].append(resi)
        for ch in sorted(by_chain):
            uniq = sorted(set(by_chain[ch]))
            for i, r in enumerate(uniq, 1):
                inv[(str(r), ch)] = i + offset
            offset += len(uniq)
        return inv

    # CIF
    by_chain = defaultdict(list)
    for a in read_cif_atoms(struct_path):
        ch = a["chain"] or "A"
        try:
            resi = int(float(a["resid"]))
        except ValueError:
            continue
        by_chain[ch].append(resi)
    for ch in sorted(by_chain):
        uniq = sorted(set(by_chain[ch]))
        for i, r in enumerate(uniq, 1):
            inv[(str(r), ch)] = i + offset
        offset += len(uniq)
    return inv

def load_itp(path):
    s = set()
    for line in open(path):
        if line.startswith("molecule_0_"):
            a, b = line.split()[:2]
            i, j = map(int, [a.rsplit("_", 1)[1], b.rsplit("_", 1)[1]])
            s.add((min(i, j), max(i, j)))
    return s

def write_mock(highfile, struct_path, itp_out):
    """
    Write a mock Go ITP using residue indices (i1, i2) and chain IDs (c1, c2).
    Works for PDB or CIF, using build_index.
    """
    inv = build_index(struct_path)  # keys: (str(resid), chain) -> sequential bead index
    with open(itp_out, "w") as out:
        out.write("[ nonbond_params ]\n")
        with open(highfile) as hf:
            next(hf, None)  # skip header
            for line in hf:
                p = line.split()
                if len(p) < 5:
                    continue
                resid1, resid2 = p[0], p[1]
                ch1, ch2 = p[3], p[4]
                i1 = inv.get((resid1, ch1))
                i2 = inv.get((resid2, ch2))
                if i1 and i2:
                    out.write(f"molecule_0_{i1} molecule_0_{i2} 1 0.00000000 0.00000000 ; mock\n")

# ---------------- main ----------------

def main():
    import sys

    # Log the command used to run the script
    with open("run.log", "a") as log_file:
        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command: {' '.join(sys.argv)}\n")

    parser = argparse.ArgumentParser(description="Run full contact analysis and build coarse-grained model")
    parser.add_argument("--short", type=float, default=3.0, help="Minimum distance cutoff in angstroms")
    parser.add_argument("--long", type=float, default=11.0, help="Maximum distance cutoff in angstroms")
    parser.add_argument("--cm", default=".", help="Path to contact_map executable directory")
    parser.add_argument("--type", choices=["both","intra","inter"], default="both", help="Contact type")
    parser.add_argument("--cpus", type=int, default=15, help="Number of parallel processes")
    parser.add_argument("--threshold", type=float, default=0.7, help="Frequency threshold for high-frequency contacts")

    # martinize2 related arguments
    parser.add_argument("--merge", type=str, default=None, help="Chains to merge or 'all'")

    # optional DSSP
    parser.add_argument("--dssp", dest="dssp_path", default=None,
                        help="Path to dssp executable (optional)")

    # position restraints
    parser.add_argument("--posres", choices=["none", "all", "backbone"], default="none",
                        help="Output position restraints (none/all/backbone)")

    # manual secondary structure
    parser.add_argument("--ss", type=str, default=None,
                        help="Manual secondary structure string or single letter (optional)")

    # go-model controls
    parser.add_argument("--go-eps", dest="go_eps", type=float, default=9.414,
                        help="Epsilon for go potential")

    # termini patches
    parser.add_argument("--nter", dest="nter", action="append", default=None,
                        help="Patch for N-termini (can be used multiple times)")
    parser.add_argument("--cter", dest="cter", action="append", default=None,
                        help="Patch for C-termini (can be used multiple times)")
    parser.add_argument("--nt", dest="neutral_termini", action="store_true",
                        help="Set neutral termini (alias for -nter NH2-ter -cter COOH-ter)")

    # source force field
    parser.add_argument("--from", dest="md_source", choices=["amber","charmm"], default="amber",
                        help="Source force field for martinize2")

    # optional: force a specific frame index
    parser.add_argument("--force-frame", type=int, default=None,
                        help="Use this specific frame index for martinize2 (bypass automatic selection)")

    args = parser.parse_args()

    # discover frames (PDB and/or CIF)
    frames_map = list_frames()
    if not frames_map:
        raise FileNotFoundError("No frames found. Expected frame_####.pdb or frame_####.cif")
    frames = [frames_map[i] for i in sorted(frames_map.keys())]

    # optional merge all chains
    if args.merge == "all" and frames:
        first = frames[0]
        if first.lower().endswith(".pdb"):
            uni = mda.Universe(first)
            chains = sorted({(seg.segid or "").strip() for seg in uni.segments if (seg.segid or "").strip()})
        else:
            chains = get_cif_chains(first)
        args.merge = ",".join(chains)

    # run external contact mapper and clean maps
    run_contact_map(frames, args.cm, args.cpus)
    clean_maps(".", "orig_maps", header_regex=r"ID\s+I1\s+AA\s+C\s+I\(PDB\)")

    # filter and annotate
    filtered = []
    for mfile in glob.glob("*.map"):
        base, _ = os.path.splitext(mfile)
        out_txt = f"filtered_{os.path.basename(base)}.txt"
        filter_map(mfile, args.short, args.long, out_txt)
        filtered.append(out_txt)

    for f in filtered:
        suffix = "_intra" if args.type == "intra" else "_inter" if args.type == "inter" else ""
        outp = f"annotated_{f.replace('.txt', suffix + '.txt')}"
        annotate(f, outp,
                 keep_same=(args.type in ("both", "intra")),
                 keep_diff=(args.type in ("both", "inter")))

    # frequency over all annotated files
    norm_file = f"normalized_{args.type}.txt"
    high_file = f"high_{args.type}.txt"
    analyze_frequency("annotated_*.txt", norm_file, high_file, args.threshold)

    # per-frame counts against the high set
    write_high_counts_per_frame(high_file, "annotated_*.txt", "high_counts_per_frame.txt")

    # determine available frame files for selection and martinize2
    available_map = {}
    for root in (".", "output_files"):
        if os.path.isdir(root):
            for path in glob.glob(os.path.join(root, "frame_*.*")):
                m = FRAME_RE.match(os.path.basename(path))
                if m:
                    available_map[int(m.group(1))] = path
    # also add discovered frames
    available_map.update(frames_map)

    # choose frame: forced or automatic among available frames
    if args.force_frame is not None:
        if args.force_frame not in available_map:
            raise FileNotFoundError(f"--force-frame {args.force_frame} has no frame file in . or output_files/")
        frame_idx = int(args.force_frame)
        atom_path = available_map[frame_idx]
    else:
        high_keys = set()
        with open(high_file) as fh:
            next(fh, None)
            for line in fh:
                k = _key_from_high_line(line)
                if k:
                    high_keys.add(k)
        candidates = []
        for fn in glob.glob("annotated_*.txt"):
            m = re.search(r"frame_(\d+)", os.path.basename(fn))
            if not m:
                continue
            idx = int(m.group(1))
            if idx not in available_map:
                continue
            cnt = 0
            with open(fn) as f:
                for L in f:
                    if not L.strip() or L.startswith("Res1"):
                        continue
                    q = _key_from_annotated_line(L)
                    if q and q in high_keys:
                        cnt += 1
            candidates.append((idx, cnt, fn))
        if not candidates:
            raise FileNotFoundError("No frame available that matches annotated_*.txt")
        max_cnt = max(c for _, c, _ in candidates)
        best_idx = min(i for i, c, _ in candidates if c == max_cnt)  # smallest index in tie
        frame_idx = best_idx
        atom_path = available_map[frame_idx]

    print(f"Using frame {frame_idx} -> {atom_path}")

    # locate the corresponding .map for the selected frame
    base = os.path.splitext(os.path.basename(atom_path))[0]  # frame_XXXX
    map_candidate_same_dir = os.path.join(os.path.dirname(atom_path) or ".", f"{base}.map")
    map_candidate_out = os.path.join("output_files", f"{base}.map")
    if os.path.isfile(map_candidate_same_dir):
        go_map_path = map_candidate_same_dir
    elif os.path.isfile(map_candidate_out):
        go_map_path = map_candidate_out
    else:
        raise FileNotFoundError(f"Map file for selected frame not found: {base}.map")

    # run martinize2 with -f pointing to PDB or CIF, as present
    atom_path = run_martinize_from_atom(
        atom_path,
        go_map_path,
        args.merge,
        args.dssp_path,         # may be None
        args.go_eps,
        args.md_source,
        args.posres,
        args.ss,
        args.nter,
        args.cter,
        args.neutral_termini
    )

    # build index and reverse map from the selected structure (PDB or CIF)
    inv_map = build_index(atom_path)                          # (resid_str, chain) -> seq_idx
    inv_rev_full = {seq_idx: key for key, seq_idx in inv_map.items()}  # seq_idx -> (resid_str, chain)
    inv_map_inv = {v: k[1] for k, v in inv_map.items()}      # seq_idx -> chain

    # collect high-frequency pairs mapped into sequential bead indices (for filtering)
    high_pairs = set()
    with open(high_file) as hf:
        next(hf)
        for line in hf:
            p = line.split()
            if len(p) < 5:
                continue
            # p: i1 i2 freq c1 c2 r1 r2
            resid1, resid2 = p[0], p[1]
            ch1, ch2 = p[3], p[4]
            i1 = inv_map.get((resid1, ch1))
            i2 = inv_map.get((resid2, ch2))
            if i1 and i2:
                high_pairs.add((min(i1, i2), max(i1, i2)))

    # write mock using residue indices and chains
    mock_itp = f"go_nbparams_mock_{args.type}.itp"
    write_mock(high_file, atom_path, mock_itp)

    # --- rewrite go_nbparams.itp with proper header and filtering ---
    src_itp = "go_nbparams.itp"
    bak_itp = "go_nbparams.itp.bak"
    shutil.copy(src_itp, bak_itp)

    header_re = re.compile(r'^\s*\[\s*nonbond_params\s*\]\s*$', re.IGNORECASE)

    with open(bak_itp, "r") as rf, open(src_itp, "w") as wf:
        # always write exactly one header
        wf.write("[ nonbond_params ]\n")

        for line in rf:
            ls = line.strip()

            # skip any existing section headers to avoid duplicates
            if header_re.match(ls):
                continue

            # pass through comments and blanks unchanged
            if not ls or ls.startswith(";"):
                wf.write(line)
                continue

            # process only pair lines; pass through anything else
            if not ls.startswith("molecule_0_"):
                wf.write(line)
                continue

            # parse bead indices
            try:
                a, b = ls.split()[:2]
                i1 = int(a.rsplit("_", 1)[1])
                i2 = int(b.rsplit("_", 1)[1])
            except Exception:
                wf.write(line)
                continue

            idx_pair = (min(i1, i2), max(i1, i2))

            # chain labeling from atomistic mapping
            same_chain = (inv_map_inv.get(i1) == inv_map_inv.get(i2))

            # filtering policy:
            # inter  -> keep all intra, and only high-frequency inter
            # intra  -> keep all intra, and only high-frequency inter
            # both   -> keep only high-frequency (intra and inter)
            if args.type == "inter":
                keep = same_chain or ((not same_chain) and (idx_pair in high_pairs))
            elif args.type == "intra":
                keep = same_chain or (idx_pair in high_pairs)
            else:  # both
                keep = (idx_pair in high_pairs)

            if keep:
                wf.write(line)

    print("ITP filtering done:",
          f"type={args.type}, high_pairs={len(high_pairs)}",
          flush=True)

    # measure distances for missing high-frequency pairs and write a separate ITP
    mock_pairs = load_itp(mock_itp)
    real_pairs_after = load_itp("go_nbparams.itp")
    missing = mock_pairs - real_pairs_after

    # prepare mapping info for missing
    missing_info = []
    with open(high_file) as hf:
        next(hf, None)
        for line in hf:
            p = line.split()
            if len(p) < 5:
                continue
            r1_resid, r2_resid = p[0], p[1]
            ch1, ch2 = p[3], p[4]
            i1 = inv_map.get((r1_resid, ch1))
            i2 = inv_map.get((r2_resid, ch2))
            if i1 and i2 and (min(i1, i2), max(i1, i2)) in missing:
                missing_info.append((r1_resid, ch1, r2_resid, ch2))

    # collect all frame files for distance measurement (PDB and CIF)
    frame_files = sorted(set(glob.glob("frame_*.pdb") + glob.glob("frame_*.cif")))

    dist_dict = {mi: [] for mi in missing_info}
    for fpath in tqdm(frame_files, desc="Measuring missing distances"):
        if fpath.lower().endswith(".pdb"):
            # MDAnalysis path for PDB only
            u = mda.Universe(fpath)
            for r1, c1, r2, c2 in missing_info:
                sel1 = u.select_atoms(f"segid {c1} and resid {r1} and name CA")
                sel2 = u.select_atoms(f"segid {c2} and resid {r2} and name CA")
                if sel1 and sel2:
                    # MDAnalysis is in angstroms; convert to nm
                    d_nm = distance_array(sel1.positions, sel2.positions)[0, 0] / 10.0
                    dist_dict[(r1, c1, r2, c2)].append(d_nm)
        else:
            # CIF path: use lightweight parser
            coords = get_cif_ca_coords(fpath)  # angstroms
            for r1, c1, r2, c2 in missing_info:
                k1 = (r1, c1); k2 = (r2, c2)
                if k1 in coords and k2 in coords:
                    d_ang = np.linalg.norm(coords[k1] - coords[k2])
                    dist_dict[(r1, c1, r2, c2)].append(d_ang / 10.0)  # nm

    missing_itp = "missing_high_freq.itp"
    with open(missing_itp, "w") as wf:
        wf.write("; missing high-frequency contacts\n")
        for (r1, c1, r2, c2), ds in dist_dict.items():
            if not ds:
                continue
            avg = np.mean(ds)
            rmin = avg / (2 ** (1 / 6))
            i1, i2 = inv_map[(r1, c1)], inv_map[(r2, c2)]
            wf.write(f"molecule_0_{i1} molecule_0_{i2} 1 {rmin:.8f} {args.go_eps:.8f} ; go bond {avg:.4f}\n")

    # build reference sets for per-frame counting
    high_ref = set()
    with open(high_file) as fh:
        next(fh, None)
        for line in fh:
            k = _key_from_high_line(line)
            if k:
                high_ref.add(k)

    go_ref = go_pairs_as_resid_chain("go_nbparams.itp", inv_rev_full)

    write_counts_per_frame(high_ref, "annotated_*.txt", "high_counts_per_frame.txt", label="HighContacts")
    write_counts_per_frame(go_ref, "annotated_*.txt", "go_counts_per_frame.txt", label="GoContacts")

    # move outputs
    outdir = "output_files"
    os.makedirs(outdir, exist_ok=True)

    # move text and maps
    for ext in ("*.txt", "*.map"):
        for fn in glob.glob(ext):
            shutil.move(fn, os.path.join(outdir, fn))

    # move frames used by this run, preserving original extensions
    for path in frames:
        base = os.path.basename(path)
        if base.endswith("_CG.pdb"):
            continue
        if os.path.exists(path):
            shutil.move(path, os.path.join(outdir, base))

    print("Done.")

if __name__ == "__main__":
    main()

