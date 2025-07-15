#!/usr/bin/env python3
"""
Comprehensive contact analysis pipeline including martinize2.

This script performs the following steps:
  1. Generate contact maps for each PDB frame
  2. Clean and filter contacts by distance and flags
  3. Annotate intra and inter chain contacts
  4. Compute contact frequencies and identify high frequency pairs
  5. Select the single reference frame with the most high frequency contacts
  6. Run martinize2 to build coarse grained topology and structure
  7. Build bead index, write mock ITP and filter real ITP based on type
  8. Measure distances for missing contacts and optionally append to go_nbparams.itp
  9. Rebuild go_nbparams.itp by filtering old bonds and appending high frequency ones at end
 10. Move final .txt, .pdb and .map outputs into an output_files folder

Usage:
  python contact_calculation.py [options]
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
    pdb_file, cm_dir = args
    exe = os.path.join(cm_dir, "contact_map")
    out_map = pdb_file.replace(".pdb", ".map")
    subprocess.run([exe, pdb_file], stdout=open(out_map, "w"), stderr=subprocess.DEVNULL)


def run_contact_map(pdbs, cm_dir, cpus):
    with Pool(cpus) as pool:
        for _ in tqdm(pool.imap_unordered(process_contact_map, [(p, cm_dir) for p in pdbs]),
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
        for line in f:
            if not line.startswith("R"): continue
            parts = line.split()
            try:
                i1, i2 = int(parts[5]), int(parts[9])
                dist = float(parts[10])
                flags = " ".join(parts[11:15])
                r1, c1 = parts[3], parts[4]
                r2, c2 = parts[7], parts[8]
            except (IndexError, ValueError):
                continue
            if abs(i2 - i1) >= 4 and short <= dist <= long and (ov.search(flags) or rz.search(flags)):
                out.write(f"{r1}\t{c1}\t{i1}\t{r2}\t{c2}\t{i2}\t{dist:.4f}\t{flags}\n")


def annotate(inp, outp, keep_same, keep_diff):
    seen = set()
    with open(inp) as fin, open(outp, "w") as out:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) < 7: continue
            ch1, ch2 = cols[1], cols[4]
            if (keep_same and ch1 == ch2) or (keep_diff and ch1 != ch2):
                pair = (ch1, cols[2], ch2, cols[5])
                inv  = (ch2, cols[5], ch1, cols[2])
                if pair not in seen and inv not in seen:
                    seen.add(pair)
                    rel = "same_chain" if ch1 == ch2 else "different_chains"
                    out.write(line.strip() + f"\t{rel}\n")


def analyze_frequency(pattern, out_norm, out_high, thr):
    counts = defaultdict(int)
    records = []
    for fn in glob.glob(pattern):
        lines = open(fn).read().splitlines()
        records.append([l for l in lines if l and not l.startswith("Res1")])
    total = len(records)
    for rec in records:
        for l in rec:
            c = l.split()
            if len(c) < 6: continue
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
    for line in open(highfile).read().splitlines()[1:]:
        c = line.split(); pairs.add((c[0], c[3], c[1], c[4]))
    files = glob.glob("annotated_*.txt")
    if not files: raise FileNotFoundError("No annotated files found")
    best, bc = None, -1
    for fn in files:
        cnt = sum(1 for L in open(fn).read().splitlines() if tuple(L.split()[2:6]) in pairs)
        if cnt > bc: best, bc = fn, cnt
    match = re.search(r"frame_(\d+)", os.path.basename(best))
    if not match:
        raise ValueError(f"Cannot parse frame index from '{best}'")
    return match.group(1), best


def run_martinize(idx, pdb_dir, merge, dssp, goeps, src):
    atom = f"{pdb_dir}/frame_{idx}.pdb"; cg = f"{pdb_dir}/frame_{idx}_CG.pdb"
    cmd = ["martinize2","-f",atom,"-o","topol.top","-x",cg,
           "-dssp",dssp,"-p","backbone","-ff","martini3001",
           "-cys","auto","-ignh","-from",src,
           "-go","-name","molecule_0","-go-eps",str(goeps),"-maxwarn","100"]
    if merge: cmd[5:5] = ["-merge", merge]
    subprocess.run(cmd, check=True)
    return atom


def build_index(pdb):
    by_chain = defaultdict(list)
    for l in open(pdb):
        if l.startswith(("ATOM","HETATM")):
            by_chain[l[21]].append(int(l[22:26]))
    inv, off = {}, 0
    for ch in sorted(by_chain):
        uniq = sorted(set(by_chain[ch]))
        for i, r in enumerate(uniq, 1): inv[(str(r), ch)] = i + off
        off += len(uniq)
    return inv


def write_mock(highfile, pdb, itp_out):
    inv = build_index(pdb)
    with open(highfile) as hf, open(itp_out, "w") as out:
        out.write("[ nonbond_params ]\n")
        next(hf)
        for line in hf:
            c = line.split()
            i1, i2 = inv.get((c[0],c[3])), inv.get((c[1],c[4]))
            if i1 and i2:
                out.write(f"molecule_0_{i1} molecule_0_{i2} 1 0.00000000 0.00000000 ; mock\n")


def load_itp(path):
    s = set()
    for line in open(path):
        if line.lstrip().startswith("molecule_0_"):
            a,b = line.split()[:2]; i,j = map(int,[a.rsplit("_",1)[1],b.rsplit("_",1)[1]])
            s.add((min(i,j),max(i,j)))
    return s


def main():
    parser = argparse.ArgumentParser(description="Full contact analysis and CG build set up")
    parser.add_argument("--short",    type=float, default=3.0)
    parser.add_argument("--long",     type=float, default=11.0)
    parser.add_argument("--cm",       default=".")
    parser.add_argument("--type",     choices=["both","intra","inter"], default="both")
    parser.add_argument("--cpus",     type=int, default=15)
    parser.add_argument("--threshold",type=float, default=0.7)
    parser.add_argument("--merge",    type=str,   default=None)
    parser.add_argument("--dssp",     dest="dssp_path", default="mkdssp")
    parser.add_argument("--go-eps",   dest="go_eps", type=float, default=9.414)
    parser.add_argument("--from",     dest="md_source", choices=["amber","charmm"], default="amber")
    parser.add_argument("--all-hf",   action="store_true", help="Append missing high-frequency contacts")
    args = parser.parse_args()

    pattern = re.compile(r"^frame_(\d+)\.pdb$")
    pdbs = sorted([p for p in glob.glob("*.pdb") if pattern.match(p)], key=lambda x: int(pattern.match(x).group(1)))
    if args.merge == "all" and pdbs:
        uni = mda.Universe(pdbs[0])
        chains = sorted({seg.segid.strip() for seg in uni.segments})
        args.merge = ",".join(chains)

    run_contact_map(pdbs, args.cm, args.cpus)
    clean_maps(".", "orig_maps", "ID    I1  AA  C I(PDB)")
    filtered = []
    for m in glob.glob("*.map"):
        out_txt = "filtered_" + m.replace(".map", ".txt")
        filter_map(m, args.short, args.long, out_txt)
        filtered.append(out_txt)

    for f in filtered:
        keep_same = args.type in ("both","intra")
        keep_diff = args.type in ("both","inter")
        suffix = "_intra" if args.type=="intra" else "_inter" if args.type=="inter" else ""
        outp = f"annotated_{f.replace('.txt', suffix + '.txt')}"
        annotate(f, outp, keep_same, keep_diff)
    norm_file = f"normalized_{args.type}.txt"; high_file = f"high_{args.type}.txt"
    analyze_frequency("annotated_*.txt", norm_file, high_file, args.threshold)

    frame_idx,_ = select_reference_frame(high_file)
    atom_pdb = run_martinize(frame_idx, ".", args.merge, args.dssp_path, args.go_eps, args.md_source)
    mock_itp = f"go_nbparams_mock_{args.type}.itp"; write_mock(high_file, atom_pdb, mock_itp)
    inv_map = build_index(atom_pdb)

    high_pairs = set()
    for line in open(high_file).read().splitlines()[1:]:
        p = line.split(); i1=inv_map.get((p[0],p[3])); i2=inv_map.get((p[1],p[4]));
        if i1 and i2: high_pairs.add((min(i1,i2), max(i1,i2)))

    shutil.copy("go_nbparams.itp", "go_nbparams.itp.bak")
    with open("go_nbparams.itp.bak") as rf, open("go_nbparams.itp","w") as wf:
        for line in rf:
            if line.lstrip().startswith("molecule_0_"):
                a,b=line.split()[:2]; x=int(a.rsplit("_",1)[1]); y=int(b.rsplit("_",1)[1]);
                same = (next(ch for (r,ch),idx in inv_map.items() if idx==x) ==
                        next(ch for (r,ch),idx in inv_map.items() if idx==y))
                if args.type=="both": keep=(min(x,y),max(x,y)) in high_pairs
                elif args.type=="inter": keep=(not same and (min(x,y),max(x,y)) in high_pairs) or same
                else: keep=(same and (min(x,y),max(x,y)) in high_pairs) or not same
                if keep: wf.write(line)
            else: wf.write(line)

    if args.all_hf:
        mock_pairs = load_itp(mock_itp); real_pairs = load_itp("go_nbparams.itp")
        missing = mock_pairs - real_pairs
        coords=[]
        for line in open(high_file).read().splitlines()[1:]:
            p=line.split(); r1,c1,r2,c2=p[0],p[3],p[1],p[4]
            i1=inv_map.get((r1,c1)); i2=inv_map.get((r2,c2))
            if i1 and i2 and (min(i1,i2),max(i1,i2)) in missing:
                coords.append((r1,c1,r2,c2))
        frames=sorted([f for f in glob.glob("frame_*.pdb") if not f.endswith("_CG.pdb")])
        dist_dict={mi:[] for mi in coords}
        for pdb in tqdm(frames, desc="Measuring missing distances"):
            uni=mda.Universe(pdb)
            for r1,c1,r2,c2 in coords:
                sel1=uni.select_atoms(f"segid {c1} and resid {r1} and name CA")
                sel2=uni.select_atoms(f"segid {c2} and resid {r2} and name CA")
                if sel1 and sel2:
                    dist_dict[(r1,c1,r2,c2)].append(distance_array(sel1.positions, sel2.positions)[0,0]/10.0)
        with open("go_nbparams.itp","a") as wf:
            wf.write("\n; appended missing high-frequency contacts\n")
            for (r1,c1,r2,c2),ds in dist_dict.items():
                if not ds: continue
                avg=sum(ds)/len(ds); rmin=avg/(2**(1/6))
                i1=inv_map[(r1,c1)]; i2=inv_map[(r2,c2)]
                wf.write(f"molecule_0_{i1} molecule_0_{i2} 1 {rmin:.8f} {args.go_eps:.8f} ; go bond {avg:.4f}\n")

    outdir="output_files"; os.makedirs(outdir,exist_ok=True)
    cg_ref=f"frame_{frame_idx}_CG.pdb"
    for fn in glob.glob("*.txt"): shutil.move(fn, os.path.join(outdir, fn))
    for fn in glob.glob("*.pdb"):
        if fn!=cg_ref: shutil.move(fn, os.path.join(outdir, fn))
    for fn in glob.glob("*.map"): shutil.move(fn, os.path.join(outdir, fn))

    # remove temporary backup files starting with '#*'
    for tmp in glob.glob('#*'):
        try:
            os.remove(tmp)
        except OSError:
            pass
            
    print("Done.")


if __name__ == "__main__":
    main()
