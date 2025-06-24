#!/usr/bin/env python3
"""
Extract frames from trajectory and write pdb by residue ranges per chain.
Then postprocess PDBs: renumber residues, assign chain IDs, and standardize residue names (CYX->CYS, HIE/HID->HIS).
Usage:
  python traj_to_pdb.py -traj trajectory.xtc -top topology.pdb \
      -ranges 1-123 124-246 247-369 -outdir output_dir -stride 1
"""
import argparse
import os
import glob
import MDAnalysis as mda
from string import ascii_uppercase
from tqdm import tqdm

def write_frame_by_ranges(universe, frame_index, residue_ranges, output_dir):
    """Extract one frame and write a PDB with specified residue ranges as chains."""
    universe.trajectory[frame_index]
    atoms = universe.atoms
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"frame_{frame_index:04d}.pdb")
    atom_serial = 1

    with open(filename, 'w') as f:
        for chain_idx, (start, end) in enumerate(residue_ranges):
            chain_id = ascii_uppercase[chain_idx % len(ascii_uppercase)]
            selection = atoms.select_atoms(f"resid {start}:{end}")
            # map original residue indices to new sequential numbering
            res_map = {res.ix: idx+1 for idx, res in enumerate(selection.residues)}

            for residue in selection.residues:
                new_resid = res_map[residue.ix]
                for atom in residue.atoms:
                    element = atom.name.strip()[0] if atom.name.strip() else 'X'
                    line = (
                        "{:<6s}{:>5d} {:<4s}{:1s}{:>3s} {:1s}"
                        "{:>4d}    "
                        "{:>8.3f}{:>8.3f}{:>8.3f}"
                        "{:>6.2f}{:>6.2f}          {:>2s}\n"
                    ).format(
                        "ATOM",
                        atom_serial,
                        atom.name.ljust(4),
                        "",
                        residue.resname,
                        chain_id,
                        new_resid,
                        atom.position[0], atom.position[1], atom.position[2],
                        1.00, 0.00,
                        element
                    )
                    f.write(line)
                    atom_serial += 1
        f.write("TER\nEND\n")
    return filename


def parse_ranges(range_strings):
    """Parse list of 'start-end' strings into tuples of ints."""
    ranges = []
    for r in range_strings:
        start, end = r.split('-')
        ranges.append((int(start), int(end)))
    return ranges


def process_pdb_file(file_path):
    """Renumber residues, assign chain ID, reset counters per chain."""
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    new_lines = []
    current_chain = 'A'
    residue_counter = 1
    prev_resnum = None

    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            resnum = line[22:26].strip()
            if resnum != prev_resnum:
                prev_resnum = resnum
                if line.startswith('ATOM') and residue_counter != 1:
                    residue_counter += 1
            new_line = (
                line[:21]
                + current_chain
                + f"{residue_counter:4d}"
                + line[26:]
            )
            new_lines.append(new_line)
        elif line.startswith('TER'):
            new_lines.append(line)
            current_chain = chr(ord(current_chain) + 1)
            residue_counter = 1
            prev_resnum = None
        else:
            new_lines.append(line)

    with open(file_path, 'w') as outfile:
        outfile.writelines(new_lines)


def standardize_residue_names(directory, extensions=None):
    """Replace nonstandard residue names across files in a directory."""
    if extensions is None:
        extensions = ['pdb', 'txt', 'map']
    repl = {'CYX': 'CYS', 'HIE': 'HIS', 'HID': 'HIS'}
    for ext in extensions:
        pattern = os.path.join(directory, f"*.{ext}")
        for fname in glob.glob(pattern):
            with open(fname, 'r') as f:
                content = f.read()
            for old, new in repl.items():
                content = content.replace(old, new)
            with open(fname, 'w') as f:
                f.write(content)

def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory frames to pdb by chain ranges, then postprocess."
    )
    parser.add_argument('-traj', '--trajectory', required=True, help='Trajectory file (xtc, dcd)')
    parser.add_argument('-top', '--topology', required=True, help='Topology file (pdb, gro)')
    parser.add_argument('-ranges', nargs='+', required=True, help='Residue ranges per chain (start-end)')
    parser.add_argument('-outdir', default='.', help='Output directory')
    parser.add_argument('-stride', type=int, default=1, help='Frame stride')
    args = parser.parse_args()

    residue_ranges = parse_ranges(args.ranges)
    u = mda.Universe(args.topology, args.trajectory)
    total_frames = len(u.trajectory)
    frame_indices = range(0, total_frames, args.stride)

    pdb_files = []
    for idx in tqdm(frame_indices, desc='Writing pdb frames'):
        pdb = write_frame_by_ranges(u, idx, residue_ranges, args.outdir)
        pdb_files.append(pdb)

    # postprocess each PDB: renumber and chain-id fix
    for pdb in pdb_files:
        process_pdb_file(pdb)

    # standardize residue names in all outputs
    standardize_residue_names(args.outdir)

    print(f"Processed {len(pdb_files)} frames into {args.outdir}")


if __name__ == '__main__':
    main()
