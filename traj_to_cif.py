#!/usr/bin/env python3
"""
Extract frames from a trajectory and write mmCIF by residue ranges per chain.
Postprocessing rules:
  - Hydrogens removed by default (use --keepH to keep them)
  - Residue renames: CYX -> CYS
  - Atom names dropped entirely: CY, OY, NT, CAY, CAT
  - Isoleucine atom 'CD' standardized to 'CD1'
  - Chains assigned as A..Z, AA, AB, AC, ... (unbounded)
  - Residues renumbered per chain starting from 1

Usage:
  python traj_to_cif.py --trajectory traj.xtc --topology top.pdb \
    --ranges 1-123,124-246,247-369 --outdir out --stride 1 [--keepH]
"""

import argparse
import os
import glob
from string import ascii_uppercase

import MDAnalysis as mda
from tqdm import tqdm

# --- configuration ---

RESNAME_MAP = {
    "CYX": "CYS",
    # "HSE": "HIS", "HSD": "HIS", "HID": "HIS", "HIE": "HIS", "HSP": "HIS",
}

ATOMNAME_DROP = {"CY", "OY", "NT", "CAY", "CAT"}


# --- helpers ---

def chain_name(idx: int) -> str:
    """Return Excel-like chain names: A..Z, AA, AB, ..., unbounded."""
    letters = ascii_uppercase
    name = ""
    i = idx
    while True:
        name = letters[i % 26] + name
        i //= 26
        if i == 0:
            break
        i -= 1  # carry adjustment
    return name


def is_hydrogen(atom) -> bool:
    try:
        element = (atom.element or "").strip()
    except Exception:
        element = ""
    if element.upper() == "H":
        return True
    return atom.name.strip().upper().startswith("H")


def infer_element(atom) -> str:
    try:
        element = (atom.element or "").strip()
    except Exception:
        element = ""
    if not element:
        nm = atom.name.strip()
        if not nm:
            return "X"
        if nm[0].isdigit() and len(nm) >= 2:
            return nm[1].upper()
        return nm[0].upper()
    return element.upper()[:2]


def atom_core(name_field_4: str) -> str:
    return name_field_4.strip().upper()


def standardize_resname_3(resname_3: str) -> str:
    key = resname_3.strip().upper()
    return RESNAME_MAP.get(key, key)


def ile_fix(core: str, resname_std: str) -> str:
    if resname_std == "ILE" and core == "CD":
        return "CD1"
    return core


def parse_ranges(ranges_string):
    ranges = []
    for r in ranges_string.split(','):
        start, end = r.split('-')
        ranges.append((int(start), int(end)))
    return ranges


# --- CIF writer ---

def write_cif_for_frame(universe, frame_index, residue_ranges, outdir, keep_h=False):
    universe.trajectory[frame_index]
    atoms = universe.atoms
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"frame_{frame_index:04d}.cif")

    rows = []
    atom_serial = 1

    for chain_idx, (start, end) in enumerate(residue_ranges):
        chain_id = chain_name(chain_idx)  # A..Z, AA, AB, ...
        sel = atoms.select_atoms(f"resid {start}:{end}")

        res_map = {res.resindex: idx + 1 for idx, res in enumerate(sel.residues)}

        for residue in sel.residues:
            resname_std = standardize_resname_3(residue.resname[:3])
            new_resid = res_map[residue.resindex]

            for atom in residue.atoms:
                if not keep_h and is_hydrogen(atom):
                    continue

                core = atom_core(atom.name)
                if core in ATOMNAME_DROP:
                    continue
                core = ile_fix(core, resname_std)

                element = infer_element(atom)

                rows.append({
                    "group_PDB": "ATOM",
                    "id": atom_serial,
                    "type_symbol": element,
                    "label_atom_id": core,
                    "label_alt_id": ".",
                    "label_comp_id": resname_std,
                    "label_asym_id": chain_id,
                    "label_entity_id": 1,
                    "label_seq_id": new_resid,
                    "pdbx_PDB_ins_code": "?",
                    "Cartn_x": atom.position[0],
                    "Cartn_y": atom.position[1],
                    "Cartn_z": atom.position[2],
                    "occupancy": 1.00,
                    "B_iso_or_equiv": 0.00,
                    "pdbx_formal_charge": "?",
                    "auth_atom_id": core,
                    "auth_comp_id": resname_std,
                    "auth_asym_id": chain_id,
                    "auth_seq_id": new_resid,
                    "pdbx_PDB_model_num": 1,
                })
                atom_serial += 1

    with open(fn, "w") as f:
        datablock = f"data_frame_{frame_index:04d}"
        f.write(f"{datablock}\n#\n")
        f.write("_audit_conform.dict_name mmcif_pdbx.dic\n")
        f.write("_audit_conform.dict_version 5.428\n")
        f.write("_audit_conform.dict_location http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v5_next.dic\n")
        f.write("#\n")

        headers = [
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge",
            "_atom_site.auth_atom_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
        f.write("loop_\n")
        for h in headers:
            f.write(h + "\n")

        for r in rows:
            f.write(
                f"{r['group_PDB']} "
                f"{r['id']} "
                f"{r['type_symbol']} "
                f"{r['label_atom_id']} "
                f"{r['label_alt_id']} "
                f"{r['label_comp_id']} "
                f"{r['label_asym_id']} "
                f"{r['label_entity_id']} "
                f"{r['label_seq_id']} "
                f"{r['pdbx_PDB_ins_code']} "
                f"{r['Cartn_x']:.3f} {r['Cartn_y']:.3f} {r['Cartn_z']:.3f} "
                f"{r['occupancy']:.2f} "
                f"{r['B_iso_or_equiv']:.2f} "
                f"{r['pdbx_formal_charge']} "
                f"{r['auth_atom_id']} "
                f"{r['auth_comp_id']} "
                f"{r['auth_asym_id']} "
                f"{r['auth_seq_id']} "
                f"{r['pdbx_PDB_model_num']}\n"
            )
        f.write("#\n")

    return fn


def standardize_text_like_files(directory, extensions=None):
    if extensions is None:
        extensions = ["txt", "map"]
    for ext in extensions:
        for fname in glob.glob(os.path.join(directory, f"*.{ext}")):
            with open(fname, "r") as f:
                content = f.read()
            for old, new in RESNAME_MAP.items():
                content = content.replace(old, new)
            with open(fname, "w") as f:
                f.write(content)


# --- main ---

def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory frames to mmCIF by chain residue ranges, applying standardization rules."
    )
    parser.add_argument("--trajectory", required=True, help="Trajectory file (xtc, dcd, trr, etc.)")
    parser.add_argument("--topology", required=True, help="Topology file (pdb, gro, psf, etc.)")
    parser.add_argument(
        "--ranges",
        required=True,
        help="Residue ranges per chain, e.g., 1-123,124-246,247-369",
    )
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument(
        "--keepH",
        action="store_true",
        help="Keep hydrogen atoms (default: remove hydrogens)",
    )
    args = parser.parse_args()

    residue_ranges = parse_ranges(args.ranges)
    u = mda.Universe(args.topology, args.trajectory)
    total_frames = len(u.trajectory)
    frame_indices = range(0, total_frames, args.stride)

    cif_files = []
    for idx in tqdm(frame_indices, desc="Writing mmCIF frames"):
        cif = write_cif_for_frame(u, idx, residue_ranges, args.outdir, keep_h=args.keepH)
        cif_files.append(cif)

    standardize_text_like_files(args.outdir)

    kept = "kept" if args.keepH else "removed"
    print(
        f"Processed {len(cif_files)} frames into {args.outdir} "
        f"(hydrogens {kept}; residues {', '.join([f'{k}->{v}' for k,v in RESNAME_MAP.items()])}; "
        f"atoms dropped: {sorted(ATOMNAME_DROP)}; ILE CD->CD1 applied; chain IDs A..Z, AA, AB, ...)"
    )


if __name__ == "__main__":
    main()

