#!/usr/bin/env python3
"""
Comprehensive contact analysis pipeline including martinize2.

This script performs the following steps:
  1. Generate contact maps for each PDB frame.
  2. Clean and filter contacts by distance and flags.
  3. Annotate intra and inter chain contacts.
  4. Compute contact frequencies and identify high-frequency pairs.
  5. Select the single reference frame with most high-frequency contacts.
  6. Run martinize2 to build coarse-grained topology and structure.
  7. Build bead index, write mock ITP and filter real ITP.
  8. Measure distances for missing contacts and append to ITP.
  9. Move final .txt, .pdb and .map outputs into an output_files folder.

Usage:
  python contact_analysis.py [options]

Options:
  --short FLOAT      Minimum distance cutoff (default 3.0 Å)
  --long FLOAT       Maximum distance cutoff (default 11.0 Å)
  --cm DIR           Path to contact_map executable (default .)
  --type MODE        Contact type: both, intra or inter (default both)
  --cpus INT         Number of parallel processes (default 15)
  --threshold FLOAT  Frequency threshold for high-frequency contacts (default 0.7)
  --merge STR        Comma-separated list of chains to merge, or "all" to merge all chains
  --dssp PATH        Path to dssp executable (default mkdssp)
  --go-eps FLOAT     Epsilon value for go potential (default 9.414)
  --from SOURCE      Source force field for martinize2: amber or charmm (default amber)

Note:
  martinize2 depends on the vermouth package: `pip install vermouth`.
  Merge chains must be comma-separated, e.g. A,B,C, or use "all".
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


def process_contact_map(args):
    pdb, cm = args
    exe = os.path.join(cm, "contact_map")
    out_map = pdb.replace(".pdb", ".map")
    subprocess.run([exe, pdb],
                   stdout=open(out_map, "w"),
                   stderr=subprocess.DEVNULL)


def run_contact_map(pdbs, cm, cpus):
    with Pool(cpus) as pool:
        for _ in tqdm(pool.imap_unordered(process_contact_map, [(p, cm) for p in pdbs]),
                      total=len(pdbs), desc="Mapping"):
            pass


def clean_maps(src, backup, header):
    os.makedirs(backup, exist_ok=True)
    for m in glob.glob(os.path.join(src, "*.map")):
        bkp = os.path.join(backup, os.path.basename(m))
        shutil.move(m, bkp)
        with open(bkp) as inp, open(m, "w") as out:
            hit = False
            for line in inp:
                if hit:
                    if "UNMAPPED" not in line:
                        out.write(line)
                elif header in line:
                    hit = True
                    out.write(line)


def filter_map(map_file, short, long, out_txt):
    ov = re.compile(r"1 [01] [01] [01]")
    rz = re.compile(r"[01] [01] [01] 1")
    with open(map_file) as f, open(out_txt, "w") as out:
        for L in f:
            if not L.startswith("R"):
                continue
            parts = L.split()
            try:
                i1, i2 = int(parts[5]), int(parts[9])
                d = float(parts[10])
                flags = " ".join(parts[11:15])
                r1, c1 = parts[3], parts[4]
                r2, c2 = parts[7], parts[8]
            except (IndexError, ValueError):
                continue
            if abs(i2 - i1) >= 4 and short <= d <= long and (ov.search(flags) or rz.search(flags)):
                out.write(f"{r1}\t{c1}\t{i1}\t{r2}\t{c2}\t{i2}\t{d:.4f}\t{flags}\n")


def annotate(inp, outp, keep_same, keep_diff):
    seen = set()
    with open(inp) as fin, open(outp, "w") as out:
        for L in fin:
            cols = L.strip().split("\t")
            if len(cols) < 7:
                continue
            ch1, ch2 = cols[1], cols[4]
            if (keep_same and ch1 == ch2) or (keep_diff and ch1 != ch2):
                pair = (ch1, cols[2], ch2, cols[5])
                inv = (ch2, cols[5], ch1, cols[2])
                if pair not in seen and inv not in seen:
                    seen.add(pair)
                    rel = "same_chain" if ch1 == ch2 else "different_chains"
                    out.write(L.strip() + f"\t{rel}\n")


def analyze_frequency(pattern, out_norm, out_high, thr):
    counts = defaultdict(int)
    recs = []
    for fn in glob.glob(pattern):
        lines = open(fn).read().splitlines()
        recs.append([l for l in lines if l and not l.startswith("Res1")])
    total = len(recs)
    for rec in recs:
        for l in rec:
            c = l.split()
            if len(c) < 6:
                continue
            key = (c[2], c[5], c[0], c[3], c[1], c[4])
            counts[key] += 1
    with open(out_norm, "w") as out:
        out.write("Res1\tRes2\tFreq\tChain1\tChain2\tResname1\tResname2\n")
        for k, v in counts.items():
            out.write(f"{k[0]}\t{k[1]}\t{v/total:.2f}\t{k[4]}\t{k[5]}\t{k[2]}\t{k[3]}\n")
    with open(out_high, "w") as out:
        header = open(out_norm).readline()
        out.write(header)
        for l in open(out_norm).read().splitlines()[1:]:
            if float(l.split()[2]) >= thr:
                out.write(l + "\n")


def select_reference_frame(highfile):
    pairs = set()
    for l in open(highfile).read().splitlines()[1:]:
        c = l.split()
        pairs.add((c[0], c[3], c[1], c[4]))
    files = glob.glob("annotated_*.txt")
    if not files:
        raise FileNotFoundError("No annotated files found")
    best, best_cnt = None, -1
    for fn in files:
        cnt = sum(1 for L in open(fn).read().splitlines()
                  if tuple(L.split()[2:6]) in pairs)
        if cnt > best_cnt:
            best, best_cnt = fn, cnt
    m = re.search(r"frame_(\d+)", os.path.basename(best))
    return m.group(1), best


def run_martinize(idx, pdb_dir, merge, dssp, goeps, src):
    atom = f"{pdb_dir}/frame_{idx}.pdb"
    cg   = f"{pdb_dir}/frame_{idx}_CG.pdb"
    cmd = [
        "martinize2", "-f", atom, "-o", "topol.top", "-x", cg,
        "-dssp", dssp, "-p", "backbone", "-ff", "martini3001",
        "-cys", "auto", "-ignh", "-from", src
    ]
    if merge:
        cmd += ["-merge", merge]
    cmd += ["-go", "-name", "molecule_0", "-go-eps", str(goeps), "-maxwarn", "100"]
    print("Running martinize2:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return atom


def build_index(pdb):
    byc = defaultdict(list)
    for l in open(pdb):
        if l.startswith(("ATOM","HETATM")):
            byc[l[21]].append(int(l[22:26]))
    inv, off = {}, 0
    for ch in sorted(byc):
        uniq = sorted(set(byc[ch]))
        for i, r in enumerate(uniq, 1):
            inv[(str(r), ch)] = i + off
        off += len(uniq)
    return inv

# rest of script unchanged



def write_mock(highfile, pdb, itp_out):
    inv = build_index(pdb)
    with open(highfile) as hf, open(itp_out, "w") as out:
        out.write("[ nonbond_params ]\n")
        next(hf)
        for L in hf:
            c = L.split()
            r1, r2 = c[0], c[1]
            ch1, ch2 = c[3], c[4]
            i1, i2 = inv.get((r1, ch1)), inv.get((r2, ch2))
            if i1 and i2:
                out.write(f"molecule_0_{i1} molecule_0_{i2} "
                          "1 0.00000000 0.00000000 ; mock\n")


def load_itp(path):
    s = set()
    for L in open(path):
        if L.startswith("molecule_0_"):
            a, b = L.split()[:2]
            i, j = map(int, [a.rsplit("_",1)[1], b.rsplit("_",1)[1]])
            s.add((min(i,j), max(i,j)))
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--short",    type=float, default=3.0)
    parser.add_argument("--long",     type=float, default=11.0)
    parser.add_argument("--cm",       default=".")
    parser.add_argument("--type",     choices=["both","intra","inter"], default="both")
    parser.add_argument("--cpus",     type=int,   default=15)
    parser.add_argument("--threshold",type=float, default=0.7)
    parser.add_argument("--merge",    type=str,   default=None,
                        help="Comma-separated list of chains to merge, or 'all'")
    parser.add_argument("--dssp",     dest="dssp_path", default="mkdssp")
    parser.add_argument("--go-eps",   dest="go_eps", type=float, default=9.414)
    parser.add_argument("--from",     dest="md_source",
                        choices=["amber","charmm"], default="amber")
    args = parser.parse_args()

    # pattern match for pdb files
    pattern = re.compile(r"^frame_(\d+)\.pdb$")
    pdbs = [p for p in glob.glob("*.pdb") if pattern.match(p)]
    pdbs.sort(key=lambda x: int(pattern.match(x).group(1)))

    # handle merge='all'
    if args.merge == "all":
        if pdbs:
            # use first frame to determine chains
            uni = mda.Universe(pdbs[0])
            chains = sorted({seg.segid.strip() for seg in uni.segments})
            args.merge = ",".join(chains)
        else:
            args.merge = None

    # 1-3: contact maps and filtering
    run_contact_map(pdbs, args.cm, args.cpus)
    clean_maps(".", "orig_maps", "ID    I1  AA  C I(PDB)")
    filtered = []
    for m in glob.glob("*.map"):
        out_txt = "filtered_" + m.replace(".map", ".txt")
        filter_map(m, args.short, args.long, out_txt)
        filtered.append(out_txt)

    # 4-5: annotate & frequency
    for f in filtered:
        suffix = "_intra" if args.type == "intra" else "_inter" if args.type == "inter" else ""
        outp = f"annotated_{f.replace('.txt', suffix + '.txt')}"
        annotate(f, outp,
                 keep_same=(args.type in ("both","intra")),
                 keep_diff=(args.type in ("both","inter")))
    norm_file = f"normalized_{args.type}.txt"
    high_file = f"high_{args.type}.txt"
    analyze_frequency("annotated_*.txt", norm_file, high_file, args.threshold)

    # 6: select reference frame by frequency
    frame_idx, _ = select_reference_frame(high_file)

    # 7: coarse-grain with martinize2
    atom_pdb = run_martinize(frame_idx, ".", args.merge,
                              args.dssp_path, args.go_eps,
                              args.md_source)

    # 8-10: ITP handling and missing contacts
    mock_itp = f"go_nbparams_mock_{args.type}.itp"
    write_mock(high_file, atom_pdb, mock_itp)
    inv_map = build_index(atom_pdb)

    high_pairs = set()
    for line in open(high_file).read().splitlines():
        if line.startswith("Res1"):
            continue
        parts = line.split()
        r1, r2, ch1, ch2 = parts[0], parts[1], parts[3], parts[4]
        if args.type == "intra" and ch1 != ch2: continue
        if args.type == "inter" and ch1 == ch2: continue
        i1, i2 = inv_map.get((r1, ch1)), inv_map.get((r2, ch2))
        if i1 and i2:
            high_pairs.add((min(i1, i2), max(i1, i2)))

    shutil.copy("go_nbparams.itp", "go_nbparams.itp.bak")
    with open("go_nbparams.itp.bak") as rf, open("go_nbparams.itp", "w") as wf:
        for L in rf:
            if L.startswith("molecule_0_"):
                a, b = L.split()[:2]
                idx = (min(int(a.rsplit("_",1)[1]), int(b.rsplit("_",1)[1])),
                       max(int(a.rsplit("_",1)[1]), int(b.rsplit("_",1)[1])))
                if idx in high_pairs:
                    wf.write(L)
            else:
                wf.write(L)

    real_pairs = load_itp("go_nbparams.itp")
    mock_pairs = load_itp(mock_itp)
    missing = mock_pairs - real_pairs
    missing_info = []
    for line in open(high_file).read().splitlines():
        if line.startswith("Res1"):
            continue
        parts = line.split()
        r1, r2, ch1, ch2 = parts[0], parts[1], parts[3], parts[4]
        i1, i2 = inv_map.get((r1, ch1)), inv_map.get((r2, ch2))
        if i1 and i2 and (min(i1, i2), max(i1, i2)) in missing:
            missing_info.append((r1, ch1, r2, ch2))

    pdb_files = sorted(f for f in glob.glob("frame_*.pdb") if not f.endswith("_CG.pdb"))
    dist_dict = {mi: [] for mi in missing_info}
    for pdb in tqdm(pdb_files, desc="Measuring missing distances"):
        u = mda.Universe(pdb)
        for r1, c1, r2, c2 in missing_info:
            sel1 = u.select_atoms(f"segid {c1} and resid {r1} and name CA")
            sel2 = u.select_atoms(f"segid {c2} and resid {r2} and name CA")
            if sel1 and sel2:
                d = distance_array(sel1.positions, sel2.positions)[0,0] / 10.0
                dist_dict[(r1, c1, r2, c2)].append(d)

    with open("go_nbparams.itp", "a") as wf:
        wf.write("\n; appended missing high-frequency contacts\n")
        for (r1, c1, r2, c2), ds in dist_dict.items():
            avg = np.mean(ds)
            rmin = avg / (2 ** (1/6))
            i1, i2 = inv_map[(r1, c1)], inv_map[(r2, c2)]
            wf.write(f"molecule_0_{i1} molecule_0_{i2} 1 {rmin:.8f} {args.go_eps:.8f} ; go bond {avg:.4f}\n")

    # 11: move outputs
    outdir = "output_files"
    os.makedirs(outdir, exist_ok=True)
    for ext in ("*.txt", "*.map", "*.pdb"):
        for fn in glob.glob(ext):
            if fn.endswith("_CG.pdb"):
                continue
            shutil.move(fn, os.path.join(outdir, fn))

    print("Done.")
