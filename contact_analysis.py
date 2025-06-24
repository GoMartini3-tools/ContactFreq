#!/usr/bin/env python3
"""
Comprehensive contact analysis pipeline including martinize2.
Usage:
  python contact_analysis.py \
    --short 3.0 --long 11.0 --cm . \
    --type both|intra|inter --cpus 15 \
    [--threshold 0.7] --merge A,B,C [--all-hf] [--dssp PATH] [--go-eps EPS]

Note:
  martinize2 depends on the vermouth package. Install via `pip install vermouth`.
  Merge chains must be comma-separated (e.g. A,B,C).

Flags:
  --short      short cutoff (Å) (default: 3.0)
  --long       long cutoff (Å) (default: 11.0)
  --cm     directory containing contact_map executable
  --type       analysis type: both, intra, inter
  --cpus       number of parallel workers
  --threshold  frequency threshold for high-frequency contacts
  --merge      chain list for martinize2 merge flag (e.g. A,B,C)
  --all-hf     measure and append all high-frequency contacts
  --dssp       path to DSSP executable (default: mkdssp)
  --go-eps     value for martinize2 -go-eps (default: 9.414)
"""

import os
import glob
import shutil
import re
import argparse
import subprocess
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

# Step 1: run contact_map
def process_contact_map(args):
    pdb_file, cm = args
    exe = os.path.join(cm, 'contact_map')
    out_map = pdb_file.replace('.pdb', '.map')
    subprocess.run([exe, pdb_file], stdout=open(out_map, 'w'), stderr=subprocess.DEVNULL)

def run_contact_map(pdbs, cm, cpus):
    tasks = [(p, cm) for p in pdbs]
    with Pool(cpus) as pool:
        for _ in tqdm(pool.imap_unordered(process_contact_map, tasks), total=len(tasks), desc='Mapping'):
            pass

# Step 2: clean .map files
def clean_maps(src_dir, backup_dir, header_line):
    os.makedirs(backup_dir, exist_ok=True)
    for m in glob.glob(os.path.join(src_dir, '*.map')):
        bkp = os.path.join(backup_dir, os.path.basename(m))
        shutil.move(m, bkp)
        with open(bkp) as inp, open(m, 'w') as out:
            found = False
            for line in inp:
                if found and 'UNMAPPED' not in line:
                    out.write(line)
                elif header_line in line:
                    found = True
                    out.write(line)

# Step 3: filter .map -> .txt
def filter_map(map_file, short_cut, long_cut, out_txt):
    ov_re = re.compile(r'1 [01] [01] [01]')
    rcsu_re = re.compile(r'[01] [01] [01] 1')
    with open(map_file) as f, open(out_txt, 'w') as w:
        for line in f:
            if line.startswith(('ID','=')) or not line.startswith('R'):
                continue
            parts = line.split()
            try:
                n1, n2 = int(parts[5]), int(parts[9])
                dist = float(parts[10])
                flags = ' '.join(parts[11:15])
                res1, ch1 = parts[3], parts[4]
                res2, ch2 = parts[7], parts[8]
            except:
                continue
            if abs(n2 - n1) >= 4 and short_cut <= dist <= long_cut and (ov_re.search(flags) or rcsu_re.search(flags)):
                w.write(f"{res1}\t{ch1}\t{n1}\t{res2}\t{ch2}\t{n2}\t{dist:.4f}\t{flags}\n")

# Step 4: annotate intra/inter
def annotate(input_txt, output_txt, keep_same, keep_diff):
    seen = set()
    with open(input_txt) as fin, open(output_txt, 'w') as out:
        for line in fin:
            cols = line.strip().split('\t')
            if len(cols) < 6:
                continue
            c1, c2 = cols[1], cols[4]
            if (keep_same and c1 == c2) or (keep_diff and c1 != c2):
                pair = (c1, cols[2], c2, cols[5])
                inv = (c2, cols[5], c1, cols[2])
                if pair not in seen and inv not in seen:
                    seen.add(pair)
                    rel = 'same_chain' if c1 == c2 else 'different_chains'
                    out.write(line.strip() + f"\t{rel}\n")

# Step 5: compute frequencies
def analyze_frequency(pattern, out_norm, out_high, threshold):
    counts = defaultdict(int)
    recs = []
    for fn in glob.glob(pattern):
        recs.append([l for l in open(fn).read().splitlines() if l and not l.startswith('Res1')])
    total = len(recs)
    for rec in recs:
        for l in rec:
            c = l.split()
            if len(c) < 6:
                continue
            key = (c[2], c[5], c[0], c[3], c[1], c[4])
            counts[key] += 1
    with open(out_norm, 'w') as out:
        out.write("Res1\tRes2\tFreq\tChain1\tChain2\tResname1\tResname2\n")
        for k, v in counts.items():
            out.write(f"{k[0]}\t{k[1]}\t{v/total:.2f}\t{k[4]}\t{k[5]}\t{k[2]}\t{k[3]}\n")
    with open(out_high, 'w') as out:
        out.write(open(out_norm).readline())
        for l in open(out_norm).read().splitlines()[1:]:
            if float(l.split()[2]) >= threshold:
                out.write(l + "\n")

# Step 6: select frame
def select_frame(highfile, prefix):
    pairs = set()
    for l in open(highfile).read().splitlines()[1:]:
        c = l.split()
        pairs.add((c[0], c[3], c[1], c[4]))
    files = glob.glob(f"*{prefix}*.txt")
    if not files:
        raise FileNotFoundError(f"No files for '{prefix}'")
    best, cnt_max = None, -1
    for fn in files:
        cnt = sum(1 for L in open(fn) if tuple(L.split()[2:6]) in pairs)
        if cnt > cnt_max:
            cnt_max, best = cnt, fn
    idx = ''.join(d for d in os.path.basename(best).split('_frame')[1] if d.isdigit())
    return idx, best

# Step 7: run martinize2
def run_martinize(frame_idx, pdb_dir, merge, dssp, go_eps):
    atom = f"{pdb_dir}/frame{frame_idx}.pdb"
    cg   = f"{pdb_dir}/frame{frame_idx}_CG.pdb"
    gm   = f"{pdb_dir}/frame{frame_idx}.map"
    cmd = [
        'martinize2', '-f', atom, '-o', 'topol.top', '-x', cg,
        '-dssp', dssp, '-p', 'backbone',
        '-ff', 'martini3001', '-cys', 'auto', '-ignh',
        '-from', 'amber', '-merge', merge,
        '-go', gm, '-go-moltype', 'molecule_0',
        '-go-eps', str(go_eps), '-maxwarn', '9'
    ]
    subprocess.run(cmd, check=True)
    return atom

# Step 8a: missing HF measurement config
HIGH_FREQ_FILE = 'high_frequency_contacts.txt'
ITP_FILE       = 'go_nbparams.itp'
PDB_PATTERN    = 'frame*.pdb'
ANNOT_PREF     = 'annotated'
N_CPUS         = 12
missing_contacts = []

def select_reference_frame():
    return select_frame(HIGH_FREQ_FILE, ANNOT_PREF)[0]

def build_index_map(frame_idx):
    pdb = f'frame{frame_idx}.pdb'
    by_chain = defaultdict(list)
    for L in open(pdb):
        if L.startswith(('ATOM','HETATM')):
            by_chain[L[21]].append(int(L[22:26]))
    inv, offset = {}, 0
    for ch in sorted(by_chain):
        uniq = sorted(set(by_chain[ch]))
        for idx, rid in enumerate(uniq,1): inv[(str(rid),ch)] = idx+offset
        offset += len(uniq)
    return inv

def load_missing(path):
    global missing_contacts
    m=[]
    for L in open(path).read().splitlines()[1:]:
        c=L.split()
        if len(c)>=5: m.append((c[0],c[3],c[1],c[4]))
    missing_contacts=m
    return m

# Step 8b: measure distances

def worker_task(pdb_path):
    u = mda.Universe(pdb_path)
    out=[]
    for r1,ch1,r2,ch2 in missing_contacts:
        sel1=u.select_atoms(f"segid {ch1} and resid {r1} and name CA")
        sel2=u.select_atoms(f"segid {ch2} and resid {r2} and name CA")
        if len(sel1)==1 and len(sel2)==1:
            d=distance_array(sel1.positions,sel2.positions)[0,0]/10.0
            out.append(((r1,ch1,r2,ch2),d))
    return out


def measure_distances_parallel(missing, pdbs):
    dd={p:[] for p in missing}
    with ProcessPoolExecutor(max_workers=N_CPUS) as ex:
        for res in tqdm(ex.map(worker_task,pdbs),total=len(pdbs),desc='Measuring'):
            for p,d in res: dd[p].append(d)
    return dd


def append_to_itp(itp,dist_dict,idx_map):
    cnt=0
    with open(itp,'a') as f:
        for (r1,ch1,r2,ch2),ds in dist_dict.items():
            if not ds: continue
            avg=np.mean(ds)
            rmin=avg/(2**(1/6))
            i1,i2=idx_map[(r1,ch1)],idx_map[(r2,ch2)]
            f.write(f"molecule_0_{i1} molecule_0_{i2} 1 {rmin:.8f} 12.00000000 ; go bond {avg:.6f}\n")
            cnt+=1
    print(f"Appended {cnt} contacts to {itp}")

# Step 9: mock & real ITP filter
def write_mock(contact_file,pdb_file,out_itp):
    inv=build_index_map(select_reference_frame())
    with open(contact_file) as hf,open(out_itp,'w') as out:
        out.write('[ nonbond_params ]\n')
        next(hf)
        for L in hf:
            c=L.split()
            r1,r2=c[0],c[1]
            ch1,ch2=c[3],c[4]
            i1,i2=inv.get((r1,ch1)),inv.get((r2,ch2))
            if i1 and i2: out.write(f"molecule_0_{i1} molecule_0_{i2} 1 0.00000000 0.00000000 ; mock\n")

def load_itp(path):
    s=set()
    for L in open(path):
        if L.startswith('molecule_0_'):
            a,b=L.split()[:2]
            i=int(a.rsplit('_',1)[1]); j=int(b.rsplit('_',1)[1])
            s.add((min(i,j),max(i,j)))
    return s

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--short',type=float,default=3.0)
    parser.add_argument('--long',type=float,default=11.0)
    parser.add_argument('--cm',dest='cm',default='.')
    parser.add_argument('--type',choices=['both','intra','inter'],default='both')
    parser.add_argument('--cpus',type=int,default=15)
    parser.add_argument('--threshold',type=float,default=0.7)
    parser.add_argument('--merge',default='A,B,C')
    parser.add_argument('--all-hf',action='store_true')
    parser.add_argument('--dssp',dest='dssp',default='mkdssp')
    parser.add_argument('--go-eps',dest='go_eps',type=float,default=9.414)
    args=parser.parse_args()

    pdbs=glob.glob('*.pdb')
    run_contact_map(pdbs,args.cm,args.cpus)
    clean_maps('.', 'orig_maps', 'ID    I1  AA  C I(PDB)')
    filtered=[]
    for m in glob.glob('*.map'):
        f='filtered_'+os.path.basename(m).replace('.map','.txt')
        filter_map(m,args.short,args.long,f)
        filtered.append(f)

    if args.type=='both':
        annotated=[]
        for f in filtered:
            out='annotated_'+f
            annotate(f,out,True,True)
            annotated.append(out)
        analyze_frequency('annotated_*.txt','normalized_contact_frequencies.txt','high_frequency_contacts.txt',args.threshold)
        contact_file='high_frequency_contacts.txt'
        merge_pref='annotated_filtered'
    else:
        suffix='_intra.txt' if args.type=='intra' else '_inter.txt'
        for f in filtered:
            annotate(f,f.replace('.txt',suffix),args.type!='inter',args.type!='intra')
        norm='norm'+suffix; high='high'+suffix
        analyze_frequency(f'*{suffix}',norm,high,args.threshold)
        contact_file=high; merge_pref='intra' if args.type=='intra' else 'inter'

    if args.all_hf:
        frame_idx=select_reference_frame()
        idx_map=build_index_map(frame_idx)
        hf_pairs=[(l.split()[0],l.split()[3],l.split()[1],l.split()[4]) for l in open('high_frequency_contacts.txt').read().splitlines()[1:]]
        existing=load_itp(ITP_FILE)
        missing_pairs=[p for p in hf_pairs if (min(idx_map[p[:2]]),max(idx_map[p[:2]])) not in existing]
        pbds_sorted=sorted(glob.glob(PDB_PATTERN))
        dd=measure_distances_parallel(missing_pairs,pbds_sorted)
        append_to_itp(ITP_FILE,dd,idx_map)
    else:
        idx,_=select_frame(contact_file,merge_pref)
        atom=run_martinize(idx,'.',args.merge,args.dssp,args.go_eps)
        write_mock(contact_file,atom,f'go_nbparams_mock_{merge_pref}.itp')
        inv_map=build_index_map(idx)
        hf_pairs=set()
        for ln in open(contact_file).read().splitlines()[1:]:
            c=ln.split(); r1,r2=c[0],c[1]; ch1, ch2=c[3],c[4]
            if args.type=='intra' and ch1!=ch2: continue
            if args.type=='inter' and ch1==ch2: continue
            i1,i2=inv_map.get((r1,ch1)), inv_map.get((r2,ch2))
            if i1 and i2: hf_pairs.add((min(i1,i2),max(i1,i2)))
        shutil.copyfile(ITP_FILE, ITP_FILE+'.bak')
        with open(ITP_FILE+'.bak') as rf, open(ITP_FILE,'w') as wf:
            for L in rf:
                if L.startswith('molecule_0_'):
                    a,b=L.split()[:2]; i,j=map(int,[a.rsplit('_',1)[1],b.rsplit('_',1)[1]])
                    if (min(i,j),max(i,j)) in hf_pairs: wf.write(L)
                else: wf.write(L)
        missing_pairs=load_itp(f'go_nbparams_mock_{merge_pref}.itp')-load_itp(ITP_FILE)
        with open(f'missing_{merge_pref}.txt','w') as wf:
            wf.write('#pairs missing\n')
            for i,j in missing_pairs: wf.write(f"{i}\t{j}\n")

    print('Done.')
