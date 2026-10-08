#!/usr/bin/env python3
# updated: 08-10-2026
"""
Comprehensive contact analysis pipeline including martinize2.

Now supports frames in PDB or CIF natively, without conversion.
If CIF frames are present, they are used directly so chain IDs are preserved.

This script performs the following steps:
  1. Generate contact maps for each frame (.pdb or .cif)
  2. Clean and filter contacts by distance and flags (distance thresholds in nm via -go-low and -go-up)
  3. Annotate intra and inter chain contacts
  4. Compute contact frequencies and identify high-frequency pairs
  5. Select the single reference frame with the most high-frequency contacts
  6. Run martinize2 to build coarse-grained topology and structure
  7. Build bead index, write mock ITP and filter real ITP
  8. Measure distances for missing contacts and write them to a separate ITP
  9. Write per-frame counts of high-frequency contacts and Go contacts
 10. Move final .txt, .map and frame files into an output_files folder

Usage:
  python contact_freq.py [options]
  e.g. python contact_freq.py -type both -merge all -dssp mkdssp -go-eps 15 -from charmm -cm /home/phoenix/software/

Options use a single dash, as in martinize2 (-dssp, -merge, -go-eps, ...).
The former double-dash spelling (--dssp, --merge, ...) is still accepted and
translated, with a deprecation note.

Run `python contact_freq.py -h` to see all available flags.
"""

import os
import sys
import shlex
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
import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="MDAnalysis.topology.PDBParser")

# Prefix used by martinize2 (-name) for the Go virtual sites: <MOLNAME>_<bead index>
MOLNAME = "molecule_0"

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
            while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
                i += 1
            continue

        # consume data rows for this loop
        while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
            tokens = lines[i].split()
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
    with open(out_map, "w") as fh:
        res = subprocess.run([exe, in_file], stdout=fh,
                             stderr=subprocess.PIPE, text=True)
    return in_file, res.returncode, (res.stderr or "").strip()[-300:]

def run_contact_map(frames, cm_dir, cpus):
    exe = os.path.join(cm_dir, "contact_map")
    if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        raise FileNotFoundError(f"contact_map executable not found or not executable: {exe} (use -cm)")
    failed = []
    with Pool(cpus) as pool:
        for in_file, rc, err in tqdm(pool.imap_unordered(
                                         process_contact_map,
                                         [(p, cm_dir) for p in frames]),
                                     total=len(frames),
                                     desc="Mapping"):
            if rc != 0:
                failed.append((in_file, rc, err))
    for in_file, rc, err in failed[:10]:
        print(f"WARNING: contact_map exited with code {rc} on {in_file}: {err}")
    if failed:
        print(f"WARNING: contact_map reported errors on {len(failed)} of {len(frames)} frames")
        if all(os.path.getsize(os.path.splitext(p)[0] + ".map") == 0 for p in frames):
            raise RuntimeError("contact_map produced empty maps for every frame; check -cm and the inputs")

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

def filter_map(map_file, low_nm, up_nm, out_txt, min_seq_sep=4):
    """
    Keep contacts with distance between low_nm and up_nm inclusive.
    Map distances are in angstroms, so thresholds in nm are converted to angstroms.
    """
    low_a = low_nm * 10.0
    up_a = up_nm * 10.0
    ov = re.compile(r"1 [01] [01] [01]")
    rz = re.compile(r"[01] [01] [01] 1")
    with open(map_file) as f, open(out_txt, "w") as out:
        for line in f:
            if not line.startswith("R"):
                continue
            parts = line.split()
            try:
                i1, i2 = int(parts[5]), int(parts[9])       # I(PDB)
                dist_a = float(parts[10])                   # angstroms from contact_map
                flags = " ".join(parts[11:15])
                r1, c1 = parts[3], parts[4]                 # resname, chain
                r2, c2 = parts[7], parts[8]
            except (IndexError, ValueError):
                continue
            if ((c1 != c2 or abs(i2 - i1) >= min_seq_sep) and
                low_a <= dist_a <= up_a and
                (ov.search(flags) or rz.search(flags))):
                out.write(f"{r1}\t{c1}\t{i1}\t{r2}\t{c2}\t{i2}\t{dist_a:.4f}\t{flags}\n")

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

def _num(x):
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)

def analyze_frequency(pattern, out_norm, out_high, thr):
    """
    Build per-pair frequency over all annotated_* files.

    Pairs are keyed by an orientation independent (resid, chain, resid, chain)
    tuple, so the same contact is counted once regardless of the order in which
    contact_map reported it, and residue names (which may vary between frames,
    e.g. HIS/HID/HIE) are not part of the key. The threshold is applied to the
    exact fraction, not to the rounded value written to disk.

    Output columns: Res1 Res2 Freq Chain1 Chain2 Resname1 Resname2
    """
    counts = defaultdict(int)
    names = {}
    files = sorted(glob.glob(pattern))
    for fn in files:
        seen = set()
        with open(fn) as fh:
            for l in fh:
                c = l.split()
                if len(c) < 6 or c[0] == "Res1":
                    continue
                a = (c[2], c[1], c[5], c[4])
                b = (c[5], c[4], c[2], c[1])
                key = a if a <= b else b
                if key in seen:
                    continue
                seen.add(key)
                counts[key] += 1
                if key not in names:
                    names[key] = (c[0], c[3]) if key == a else (c[3], c[0])
    total = len(files) or 1

    header = "Res1\tRes2\tFreq\tChain1\tChain2\tResname1\tResname2\n"
    order = sorted(counts, key=lambda k: (k[1], _num(k[0]), k[3], _num(k[2])))
    with open(out_norm, "w") as out, open(out_high, "w") as hi:
        out.write(header)
        hi.write(header)
        for k in order:
            v = counts[k]
            r1, r2 = names[k]
            line = f"{k[0]}\t{k[2]}\t{v/total:.2f}\t{k[1]}\t{k[3]}\t{r1}\t{r2}\n"
            out.write(line)
            if v / total >= thr - 1e-9:
                hi.write(line)

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
        if not line.startswith(MOLNAME + "_"):
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
                            neutral_termini,
                            *,
                            go_low,
                            go_up,
                            go_res_dist,
                            go_write_file,
                            go_backbone,
                            go_atomname,
                            water_bias,
                            water_bias_eps,
                            id_regions,
                            idr_tune,
                            noscfix,
                            scfix,
                            cys,
                            mutate,
                            modify,
                            write_graph,
                            write_repair,
                            write_canon,
                            vcount,
                            maxwarn_list,
                            to_ff=None,
                            extra_ff_dir=None,
                            extra_map_dir=None,
                            ignore=None,
                            model=None,
                            posres_fc=None,
                            extra_args=None):
    atom = atom_path
    base = os.path.splitext(os.path.basename(atom))[0]
    atom_dir = os.path.dirname(atom) or "."
    cg = os.path.join(atom_dir, f"{base}_CG.pdb")

    cmd = ["martinize2", "-f", atom]

    for grp in (merge or []):
        cmd += ["-merge", grp]
    if dssp is not None:
        # empty string means: flag without executable (martinize2 falls back to mdtraj)
        cmd += ["-dssp"] + ([dssp] if dssp else [])
    if ss:
        cmd += ["-ss", ss]
        
    # force field selection
    if to_ff:
        cmd += ["-ff", to_ff]

    # additional ff dirs
    if extra_ff_dir:
        for d in extra_ff_dir:
            cmd += ["-ff-dir", d]

    # additional map dirs
    if extra_map_dir:
        for d in extra_map_dir:
            cmd += ["-map-dir", d]    

    # Go model from external map file plus tunables
    cmd += ["-go", go_map_path, "-go-eps", str(goeps)]
    if go_low is not None:
        cmd += ["-go-low", str(go_low)]
    if go_up is not None:
        cmd += ["-go-up", str(go_up)]
    if go_res_dist is not None:
        cmd += ["-go-res-dist", str(go_res_dist)]
    if go_write_file is not None:
        if go_write_file == "":
            cmd += ["-go-write-file"]
        else:
            cmd += ["-go-write-file", go_write_file]
    if go_backbone is not None:
        cmd += ["-go-backbone", go_backbone]
    if go_atomname is not None:
        cmd += ["-go-atomname", go_atomname]

    # Water bias related
    if water_bias:
        cmd += ["-water-bias"]
    if water_bias_eps:
        cmd += ["-water-bias-eps", *water_bias_eps]
    if id_regions:
        cmd += ["-id-regions", *id_regions]
    if idr_tune:
        cmd += ["-idr-tune"]

    # Protein description and modifications
    if noscfix:
        cmd += ["-noscfix"]
    if scfix:
        cmd += ["-scfix"]
    if mutate:
        cmd += ["-mutate", *mutate]
    if modify:
        cmd += ["-modify", *modify]

    # Termini patches
    for mod in (nter_list or []):
        cmd += ["-nter", mod]
    for mod in (cter_list or []):
        cmd += ["-cter", mod]
    if neutral_termini:
        cmd += ["-nt"]

    # Debug and limits
    if write_graph:
        cmd += ["-write-graph", write_graph]
    if write_repair:
        cmd += ["-write-repair", write_repair]
    if write_canon:
        cmd += ["-write-canon", write_canon]
    if vcount and vcount > 0:
        cmd += ["-v"] * vcount

    # input selection, custom force fields, restraints, passthrough
    if ignore:
        cmd += ["-ignore", *ignore]
    if model is not None:
        cmd += ["-model", str(model)]
    if posres_fc is not None:
        cmd += ["-pf", str(posres_fc)]
    if extra_args:
        cmd += list(extra_args)

    # core output and settings
    cmd += [
        "-o", "topol.top",
        "-x", cg,
        "-p", posres,
        "-cys", "auto" if cys is None else cys,
        "-ignh",
        "-name", MOLNAME,
    ]
    if src is not None:
        cmd += ["-from", src]
        
    if maxwarn_list:
        cmd += ["-maxwarn", *[str(x) for x in maxwarn_list]]
    else:
        cmd += ["-maxwarn", "100"]

    print("Running martinize2:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return atom

# ---------------- index builder for PDB and CIF ----------------

def build_index(struct_path: str):
    """
    Map (resid_str, chain_id) -> sequential bead index.
    PDB is parsed by text. CIF is parsed with read_cif_atoms.
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
        if line.startswith(MOLNAME + "_"):
            a, b = line.split()[:2]
            i, j = map(int, [a.rsplit("_", 1)[1], b.rsplit("_", 1)[1]])
            s.add((min(i, j), max(i, j)))
    return s

def write_mock(highfile, struct_path, itp_out, inv=None):
    """
    Write a mock Go ITP using residue indices (i1, i2) and chain IDs (c1, c2).
    Works for PDB or CIF, using build_index.
    """
    if inv is None:
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
                    out.write(f"{MOLNAME}_{i1} {MOLNAME}_{i2} 1 0.00000000 0.00000000 ; mock\n")

# ---------------- distance measurement for missing contacts ----------------

def ca_table_pdb(path):
    """Return {(resid_str, chain): (x, y, z)} for CA atoms of the first model, in angstroms."""
    t = {}
    with open(path) as fh:
        for l in fh:
            if l.startswith("ENDMDL"):
                break
            if l.startswith(("ATOM", "HETATM")) and l[12:16].strip() == "CA":
                try:
                    key = (str(int(l[22:26])), l[21])
                    xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
                except ValueError:
                    continue
                t.setdefault(key, xyz)  # keep first altloc
    return t

def _missing_distances(task):
    """Worker: distances (nm) for the requested pairs present in one frame."""
    path, keys1, keys2 = task
    coords = get_cif_ca_coords(path) if path.lower().endswith(".cif") else ca_table_pdb(path)
    idx, dist = [], []
    for k, (k1, k2) in enumerate(zip(keys1, keys2)):
        p1, p2 = coords.get(k1), coords.get(k2)
        if p1 is not None and p2 is not None:
            idx.append(k)
            dist.append(float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float))) / 10.0)
    return np.array(idx, dtype=int), np.array(dist, dtype=float)

# ---------------- command line ----------------

def _normalize_legacy_flags(argv, parser):
    """
    Options use a single dash (martinize2 style). For backward compatibility the
    former double-dash spelling of any known option (e.g. --dssp, --go-eps=15)
    is translated to the single-dash form, with a one-line deprecation note.
    """
    known = {s for a in parser._actions for s in a.option_strings
             if s.startswith("-") and not s.startswith("--")}
    out, legacy = [], []
    for tok in argv:
        head, sep, tail = tok.partition("=")
        if head.startswith("--") and len(head) > 2 and head != "--help" and ("-" + head[2:]) in known:
            legacy.append(head)
            tok = "-" + head[2:] + sep + tail
        out.append(tok)
    if legacy:
        print("NOTE: double-dash options are deprecated, use the single-dash form "
              f"({', '.join(sorted(set(legacy)))}).", file=sys.stderr, flush=True)
    return out

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run full contact analysis and build coarse-grained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-cm", default=".", help="Path to contact_map executable directory")
    parser.add_argument("-type", choices=["both","intra","inter"], default="both", help="Contact type")
    parser.add_argument("-cpus", type=int, default=15, help="Number of parallel processes")
    parser.add_argument("-threshold", type=float, default=0.7, help="Frequency threshold for high-frequency contacts")

    # martinize2 related arguments
    parser.add_argument("-merge", action="append", default=None,
                        help="Chains to merge (e.g. A,B) or 'all'; may be repeated for several groups")

    # optional DSSP
    parser.add_argument("-dssp", dest="dssp_path", nargs="?", const="", default=None,
                        help="Optional. Path to dssp executable; give -dssp without a value to let martinize2 use mdtraj")

    # position restraints
    parser.add_argument("-posres", choices=["none", "all", "backbone"], default="none",
                        help="Output position restraints")

    # manual secondary structure
    parser.add_argument("-ss", type=str, default=None, help="Manual secondary structure string or single letter")

    # Go model controls and contact thresholds in nm
    parser.add_argument("-go-eps", dest="go_eps", type=float, default=9.414, help="Epsilon for go potential")
    parser.add_argument("-go-low", dest="go_low", type=float, default=0.3,
                        help="Minimum contact distance threshold in nm")
    parser.add_argument("-go-up", dest="go_up", type=float, default=1.1,
                        help="Maximum contact distance threshold in nm")
    parser.add_argument("-go-res-dist", dest="go_res_dist", type=int, default=None,
                        help="Minimum graph distance below which contacts are removed")
    parser.add_argument("-go-write-file", dest="go_write_file", nargs="?", const="", default=None,
                        help="Write contact map when Martinize2 calculates it; optional output path")
    parser.add_argument("-go-backbone", dest="go_backbone", type=str, default="BB",
                        help="Backbone bead name for Go site")
    parser.add_argument("-go-atomname", dest="go_atomname", type=str, default="CA",
                        help="Virtual Go site atom name")
                        
    parser.add_argument("-ff", dest="to_ff", default="martini3001",
                    help="Coarse-grained force field for martinize2")

    parser.add_argument("-ff-dir", dest="extra_ff_dir", nargs="+", default=None,
                    help="Additional repository paths for custom force fields")

    parser.add_argument("-map-dir", dest="extra_map_dir", nargs="+", default=None,
                    help="Additional repository paths for mapping files")


    # Water bias options
    parser.add_argument("-water-bias", dest="water_bias", action="store_true",
                        help="Apply water bias to secondary structure elements")
    parser.add_argument("-water-bias-eps", dest="water_bias_eps", nargs="+", default=None,
                        help="Water bias strengths like H:3.6 C:2.1 idr:2.1")
    parser.add_argument("-id-regions", dest="id_regions", nargs="+", default=None,
                        help="Disordered regions as [chain-]start:end tokens")
    parser.add_argument("-idr-tune", dest="idr_tune", action="store_true",
                        help="Tune IDR regions with specific bonded potentials (deprecated)")

    # Protein description / modifications
    parser.add_argument("-noscfix", dest="noscfix", action="store_true",
                        help="Do not apply side chain corrections")
    parser.add_argument("-scfix", dest="scfix", action="store_true",
                        help="Legacy scfix flag")
    parser.add_argument("-cys", dest="cys", default=None, help="Cystein bonds setting")
    parser.add_argument("-mutate", dest="mutate", nargs="+", default=None,
                        help="Mutations like A-PHE45:ALA PHE30:ALA")
    parser.add_argument("-modify", dest="modify", nargs="+", default=None,
                        help="Residue modifications like A-ASP45:ASP0 ASP:ASP0 +HSE")

    # Termini patches
    parser.add_argument("-nter", dest="nter", action="append", default=None,
                        help="Patch for N-termini")
    parser.add_argument("-cter", dest="cter", action="append", default=None,
                        help="Patch for C-termini")
    parser.add_argument("-nt", dest="neutral_termini", action="store_true",
                        help="Set neutral termini")

    # source force field
    parser.add_argument("-from", dest="md_source", choices=["amber","charmm"], default=None,
                        help="Source force field for martinize2")

    # Debugging / diagnostics passthrough
    parser.add_argument("-write-graph", dest="write_graph", default=None, help="Write graph after MakeBonds")
    parser.add_argument("-write-repair", dest="write_repair", default=None, help="Write graph after RepairGraph")
    parser.add_argument("-write-canon", dest="write_canon", default=None, help="Write graph after CanonicalizeModifications")
    parser.add_argument("-v", dest="vcount", action="count", default=0, help="Increase Martinize2 verbosity")
    parser.add_argument("-maxwarn", dest="maxwarn_list", nargs="+", default=None,
                        help="Maximum allowed warnings for Martinize2")

    # Additional martinize2 passthrough and script-level options
    parser.add_argument("-ignore", dest="ignore", nargs="+", default=None,
                        help="Residue names martinize2 should ignore, e.g. HOH LIG")
    parser.add_argument("-model", type=int, default=None, help="MODEL number to read (multi-model PDB)")
    parser.add_argument("-posres-fc", dest="posres_fc", type=float, default=None,
                        help="Position restraint force constant in kJ/mol/nm^2 (martinize2 -pf)")
    parser.add_argument("-min-seq-sep", dest="min_seq_sep", type=int, default=4,
                        help="Minimum residue separation for intra-chain contacts in the frequency analysis "
                             "(not equivalent to martinize2 -go-res-dist, which is a graph distance)")
    parser.add_argument("-martinize-extra", dest="martinize_extra", default="",
                        help='Extra martinize2 flags as one string, use the = form, e.g. -martinize-extra="-bonds-fudge 1.4"')

    # Append missing high-frequency contacts
    parser.add_argument("-add-missing", dest="add_missing", action="store_true",
                        help="Append entries from missing_high_freq.itp into go_nbparams.itp to include all high-frequency contacts")

    # optional: force a specific frame index
    parser.add_argument("-force-frame", type=int, default=None,
                        help="Use this specific frame index for martinize2")

    # NEW FLAG: sigma recalculation
    parser.add_argument("-sigma", dest="sigma", action="store_true",
                        help="Recalculate sigma values from selected frame and replace them in go_nbparams.itp")

    args = parser.parse_args(_normalize_legacy_flags(sys.argv[1:], parser))

    if args.dssp_path is not None and args.ss:
        parser.error("-dssp and -ss are mutually exclusive")

    # Log the command used to run the script
    with open("run.log", "a") as log_file:
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"[{stamp}] Command: {' '.join(sys.argv)}\n")
        try:
            ver = subprocess.run(["martinize2", "-V"], capture_output=True, text=True, timeout=60)
            log_file.write(f"[{stamp}] martinize2 version: {(ver.stdout or ver.stderr).strip()}\n")
        except (OSError, subprocess.SubprocessError):
            log_file.write(f"[{stamp}] martinize2 version: unavailable\n")

    if args.dssp_path is None and not args.ss:
        print("NOTE: neither -dssp nor -ss given; martinize2 will build the topology without "
              "secondary structure information (warning suppressed by -maxwarn).", flush=True)

    # discover frames
    frames_map = list_frames()
    if not frames_map:
        raise FileNotFoundError("No frames found. Expected frame_####.pdb or frame_####.cif")
    frames = [frames_map[i] for i in sorted(frames_map.keys())]

    # optional merge all chains
    if args.merge == ["all"] and frames:
        first = frames[0]
        if first.lower().endswith(".pdb"):
            uni = mda.Universe(first)
            chains = sorted({(seg.segid or "").strip() for seg in uni.segments if (seg.segid or "").strip()})
        else:
            chains = get_cif_chains(first)
        args.merge = [",".join(chains)] if chains else None

    # run external contact mapper and clean maps
    run_contact_map(frames, args.cm, args.cpus)
    clean_maps(".", "orig_maps", header_regex=r"ID\s+I1\s+AA\s+C\s+I\(PDB\)")

    # filter and annotate using go-low and go-up in nm
    filtered = []
    for mfile in glob.glob("*.map"):
        base, _ = os.path.splitext(mfile)
        out_txt = f"filtered_{os.path.basename(base)}.txt"
        filter_map(mfile, args.go_low, args.go_up, out_txt, args.min_seq_sep)
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
    available_map.update(frames_map)

    # choose frame
    if args.force_frame is not None:
        if args.force_frame not in available_map:
            raise FileNotFoundError(f"-force-frame {args.force_frame} has no frame file in . or output_files/")
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
        best_idx = min(i for i, c, _ in candidates if c == max_cnt)
        frame_idx = best_idx
        atom_path = available_map[frame_idx]

    print(f"Using frame {frame_idx} -> {atom_path}")

    # locate the corresponding .map for the selected frame
    base = os.path.splitext(os.path.basename(atom_path))[0]
    map_candidate_same_dir = os.path.join(os.path.dirname(atom_path) or ".", f"{base}.map")
    map_candidate_out = os.path.join("output_files", f"{base}.map")
    if os.path.isfile(map_candidate_same_dir):
        go_map_path = map_candidate_same_dir
    elif os.path.isfile(map_candidate_out):
        go_map_path = map_candidate_out
    else:
        raise FileNotFoundError(f"Map file for selected frame not found: {base}.map")

    # run martinize2
    atom_path = run_martinize_from_atom(
        atom_path,
        go_map_path,
        args.merge,
        args.dssp_path,
        args.go_eps,
        args.md_source,
        args.posres,
        args.ss,
        args.nter,
        args.cter,
        args.neutral_termini,
        go_low=args.go_low,
        go_up=args.go_up,
        go_res_dist=args.go_res_dist,
        go_write_file=args.go_write_file,
        go_backbone=args.go_backbone,
        go_atomname=args.go_atomname,
        water_bias=args.water_bias,
        water_bias_eps=args.water_bias_eps,
        id_regions=args.id_regions,
        idr_tune=args.idr_tune,
        noscfix=args.noscfix,
        scfix=args.scfix,
        cys=args.cys,
        mutate=args.mutate,
        modify=args.modify,
        write_graph=args.write_graph,
        write_repair=args.write_repair,
        write_canon=args.write_canon,
        vcount=args.vcount,
        maxwarn_list=args.maxwarn_list,
        to_ff=args.to_ff,
        extra_ff_dir=args.extra_ff_dir,
        extra_map_dir=args.extra_map_dir,
        ignore=args.ignore,
        model=args.model,
        posres_fc=args.posres_fc,
        extra_args=shlex.split(args.martinize_extra)
    )

    # build index and reverse map
    inv_map = build_index(atom_path)                          # (resid_str, chain) -> seq_idx
    inv_rev_full = {seq_idx: key for key, seq_idx in inv_map.items()}  # seq_idx -> (resid_str, chain)
    inv_map_inv = {v: k[1] for k, v in inv_map.items()}      # seq_idx -> chain

    # collect high-frequency pairs mapped into sequential bead indices
    high_pairs = set()
    with open(high_file) as hf:
        next(hf, None)
        for line in hf:
            p = line.split()
            if len(p) < 5:
                continue
            resid1, resid2 = p[0], p[1]
            ch1, ch2 = p[3], p[4]
            i1 = inv_map.get((resid1, ch1))
            i2 = inv_map.get((resid2, ch2))
            if i1 and i2:
                high_pairs.add((min(i1, i2), max(i1, i2)))

    # write mock using residue indices and chains
    mock_itp = f"go_nbparams_mock_{args.type}.itp"
    write_mock(high_file, atom_path, mock_itp, inv_map)

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
            if not ls.startswith(MOLNAME + "_"):
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

            pair_key = (min(i1, i2), max(i1, i2))
            c1 = inv_map_inv.get(i1)
            c2 = inv_map_inv.get(i2)
            same_chain = (c1 is not None and c2 is not None and c1 == c2)

            # decide which Go contacts to keep depending on type
            if args.type == "both":
                # keep only high-frequency (intra and inter)
                keep = (pair_key in high_pairs)

            elif args.type == "intra":
                if same_chain:
                    # for intra contacts: keep only high-frequency intra
                    keep = (pair_key in high_pairs)
                else:
                    # for inter contacts: do not touch them, always keep
                    keep = True

            else:  # args.type == "inter"
                if same_chain:
                    # for intra contacts: do not touch them, always keep
                    keep = True
                else:
                    # for inter contacts: keep only high-frequency inter
                    keep = (pair_key in high_pairs)

            if keep:
                wf.write(line)
        # if not keep: drop this Go pair

    print("ITP filtering done:",
          f"type={args.type}, kept_high_pairs={len(load_itp('go_nbparams.itp'))}, high_pairs_total={len(high_pairs)}",
          flush=True)

    # --- optional sigma recalculation from selected frame ---
    if getattr(args, "sigma", False):
        print("Recalculating sigma values from selected frame distances...", flush=True)

        # Load the selected structure
        u = mda.Universe(atom_path)

        # Collect current pairs from the filtered go_nbparams.itp
        pairs = []
        for line in open("go_nbparams.itp"):
            if line.startswith(MOLNAME + "_"):
                a, b = line.split()[:2]
                try:
                    i1 = int(a.rsplit("_", 1)[1])
                    i2 = int(b.rsplit("_", 1)[1])
                    pairs.append((i1, i2))
                except Exception:
                    continue

        # Measure distances on the selected frame and compute sigma = distance / 2^(1/6)
        dist_data = {}
        for i1, i2 in tqdm(pairs, desc="Computing sigma from frame"):
            (r1, c1) = inv_rev_full.get(i1, (None, None))
            (r2, c2) = inv_rev_full.get(i2, (None, None))
            if not all([r1, r2, c1, c2]):
                continue
            sel1 = u.select_atoms(f"segid {c1} and resid {r1} and name CA")
            sel2 = u.select_atoms(f"segid {c2} and resid {r2} and name CA")
            if sel1.n_atoms > 0 and sel2.n_atoms > 0:
                d_nm = distance_array(sel1.positions, sel2.positions)[0, 0] / 10.0
                sigma = d_nm / (2 ** (1 / 6))
                dist_data[(min(i1, i2), max(i1, i2))] = sigma

        # Rewrite go_nbparams.itp replacing only sigma values
        tmp_out = "go_nbparams_sigma.itp"
        with open("go_nbparams.itp", "r") as rf, open(tmp_out, "w") as wf:
            for line in rf:
                if line.startswith(MOLNAME + "_"):
                    parts = line.split()
                    if len(parts) < 5:
                        wf.write(line)
                        continue
                    a, b = parts[:2]
                    try:
                        i1 = int(a.rsplit("_", 1)[1])
                        i2 = int(b.rsplit("_", 1)[1])
                    except Exception:
                        wf.write(line)
                        continue
                    pkey = (min(i1, i2), max(i1, i2))
                    sigma = dist_data.get(pkey)
                    if sigma is not None:
                        # keep epsilon from the line if it exists, otherwise fallback to args.go_eps
                        try:
                            eps = float(parts[4])
                        except Exception:
                            eps = args.go_eps
                        wf.write(f"{a} {b} 1 {sigma:.8f} {eps:.8f} ; sigma from frame {frame_idx}\n")
                    else:
                        wf.write(line)
                else:
                    wf.write(line)

        shutil.move(tmp_out, "go_nbparams.itp")
        print("Sigma recalculation complete.", flush=True)

    # measure distances for missing high-frequency pairs and write a separate ITP
    mock_pairs = load_itp(mock_itp)
    real_pairs_after = load_itp("go_nbparams.itp")
    missing = mock_pairs - real_pairs_after

    # prepare mapping info for missing
    missing_info = []
    seen_missing = set()
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
            if i1 and i2 and (min(i1, i2), max(i1, i2)) in missing \
                    and (min(i1, i2), max(i1, i2)) not in seen_missing:
                seen_missing.add((min(i1, i2), max(i1, i2)))
                missing_info.append((r1_resid, ch1, r2_resid, ch2))

    # frames already deduplicated by index (PDB preferred); excludes *_CG.pdb outputs
    frame_files = list(frames)

    # distances are accumulated in nanometers to match Martinize2 Go parameters
    dist_avg = {}
    if missing_info:
        keys1 = [(r1, c1) for r1, c1, _, _ in missing_info]
        keys2 = [(r2, c2) for _, _, r2, c2 in missing_info]
        dist_sum = np.zeros(len(missing_info))
        dist_n = np.zeros(len(missing_info), dtype=int)
        with Pool(args.cpus) as pool:
            for idx, d_nm in tqdm(pool.imap(_missing_distances,
                                            [(f, keys1, keys2) for f in frame_files]),
                                  total=len(frame_files),
                                  desc="Measuring missing distances"):
                dist_sum[idx] += d_nm
                dist_n[idx] += 1
        dist_avg = {mi: dist_sum[k] / dist_n[k]
                    for k, mi in enumerate(missing_info) if dist_n[k] > 0}

    missing_itp = "missing_high_freq.itp"
    with open(missing_itp, "w") as wf:
        wf.write("; missing high-frequency contacts\n")
        for (r1, c1, r2, c2), avg in dist_avg.items():   # avg in nm
            if avg > args.go_up:  # keep only if within the go_up threshold (nm)
                continue
            rmin = avg / (2 ** (1 / 6))       # nm, Lennard-Jones minimum
            i1, i2 = inv_map[(r1, c1)], inv_map[(r2, c2)]
            wf.write(f"{MOLNAME}_{i1} {MOLNAME}_{i2} 1 {rmin:.8f} {args.go_eps:.8f} ; go bond {avg:.4f}\n")

    # optionally append missing high-frequency contacts into the selected ITP
    if args.add_missing and os.path.isfile(missing_itp):
        with open("go_nbparams.itp", "a") as out, open(missing_itp, "r") as addf:
            for ln in addf:
                ls = ln.strip()
                if not ls:
                    continue
                if ls.startswith(";") or ls.startswith(MOLNAME + "_"):
                    out.write(ln)
        print("Appended missing high-frequency contacts into go_nbparams.itp", flush=True)

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

    for pat in ("filtered_*.txt", "annotated_*.txt", "normalized_*.txt",
                "high_*.txt", "*_per_frame.txt", "*.map"):
        for fn in glob.glob(pat):
            shutil.move(fn, os.path.join(outdir, fn))

    for path in frames:
        base = os.path.basename(path)
        if base.endswith("_CG.pdb"):
            continue
        if os.path.exists(path):
            shutil.move(path, os.path.join(outdir, base))

    print("Done.")

if __name__ == "__main__":
    main()
