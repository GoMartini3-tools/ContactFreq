# High-frequency-contacts

Tools to compute **high-frequency native contacts** from MD trajectories and build a coarse-grained (CG) Gō-Martini model with **martinize2**. Command-line options follow the martinize2 convention (single dash).

The workflow converts a trajectory into per-frame structures (PDB/CIF), computes per-frame contact maps, identifies persistent (high-frequency) contacts, selects a reference frame, and generates CG topology/parameters. It supports **PDB or CIF** frames natively and uses **nm** for contact thresholds.

---

## Requirements

- Python 3.8+
- Packages: `MDAnalysis`, `numpy`, `tqdm`
- `contact_map` executable (from GoMartini ContactMapGenerator)
- `martinize2` in your `$PATH`
- (Optional) `mkdssp` for secondary structure

Get `contact_map`:

- Zenodo: https://zenodo.org/records/3817447
- GitHub: https://github.com/Martini-Force-Field-Initiative/GoMartini/tree/main/ContactMapGenerator

---

## Step 1 - dump trajectory frames

Use `traj_to_pdb.py` to write frames as `frame_####.pdb` (or `.cif`).

```bash
python traj_to_pdb.py   -trajectory 6ZH9_WT_dry_100.nc   -topology   6ZH9_WT_dry.parm7   -ranges     1-195,195-323   -outdir     .   -stride     1
```

**Options:**

- `-trajectory` trajectory file (e.g., `.xtc`, `.dcd`, `.nc`)
- `-topology` topology or coordinates (e.g., `.pdb`, `.gro`, `.parm7`)
- `-ranges` residue blocks per chain (e.g., `2-196,197-325`)
- `-outdir` where frames are written
- `-stride` keep every n-th frame

Frames must be named `frame_0001.pdb`, `frame_0002.pdb`, … (or `.cif`).
If both `.pdb` and `.cif` exist for the same index, PDB is preferred.

---

## Step 2 - compute high-frequency contacts and build CG model

The main script is `contact_freq.py` (formerly `contact_calculation.py`). It:

- Generates `.map` files for each frame (parallelized).
- Filters contacts using nm thresholds (`-go-low`, `-go-up`).
- Annotates intra and inter contacts and removes duplicates.
- Computes contact frequencies across frames and writes:
  - `normalized_<type>.txt`
  - `high_<type>.txt` (≥ `-threshold`)
- Selects the frame with the most high-frequency contacts.
- Runs `martinize2` with `-go <frame_X.map>` matching `-f <frame_X.pdb|cif>`.
- Produces and filters `go_nbparams.itp`.
- Computes average CA–CA distances (in nm) for high-frequency pairs missing from the selected ITP and writes `missing_high_freq.itp`.
- Keeps only pairs with average distance ≤ `-go-up`.
- (Optional) `-add-missing` appends `missing_high_freq.itp` entries (with header) into `go_nbparams.itp`.
- Writes per-frame counts for high and Gō sets and moves outputs into `output_files/`.

---

## Typical run

```bash
python contact_freq.py   -cm /path/to/contact_map   -type inter   -cpus 15   -threshold 0.7   -merge all   -dssp /usr/bin/mkdssp   -from charmm   -go-eps 15.0   -go-low 0.3   -go-up 1.1   -add-missing
```

---

## Key options (with defaults)

All options use a **single dash**, as in martinize2 (`-dssp`, `-merge`, `-go-eps`, ...), and the same convention applies to `traj_to_pdb.py` and `traj_to_cif.py`. The former double-dash spelling (`--dssp`, `--merge`, ...) is still accepted and translated, with a deprecation note, so existing commands keep working.

- `-cm` (str, `"."`): directory containing `contact_map`
- `-type` (`both|intra|inter`, `"both"`): scope of contacts  
  *For monomers, use `intra`.*
- `-cpus` (int, `15`): parallel workers for mapping
- `-threshold` (float, `0.7`): high-frequency cutoff
- `-merge` (`all|A,B|None`, `None`): chains to merge before martinize2 (`all` merges every chain). Repeatable for several groups, e.g. `-merge A,B -merge C,D`
- `-dssp` (optional, `None`): path to `mkdssp`. `-dssp` with no value lets martinize2 use mdtraj. DSSP is optional; mutually exclusive with `-ss`. If neither `-dssp` nor `-ss` is given, a note is printed and the topology is built without secondary structure information
- `-from` (`amber|charmm`, `"amber"`): source force field for martinize2
- `-posres` (`none|all|backbone`, `"none"`): position restraints

**Gō model and contact filtering (nm):**

- `-go-eps` (float, `9.414`): epsilon for Gō bias (kJ/mol)
- `-go-low` (float, `0.3`): minimum contact distance threshold
- `-go-up` (float, `1.1`): maximum contact distance threshold
- `-go-res-dist` (int|None, `None`): remove contacts below a graph distance
- `-go-backbone` (str, `"BB"`): backbone bead name for Gō site
- `-go-atomname` (str, `"CA"`): virtual site atom name
- `-go-write-file` (flag or str): ask martinize2 to write its contact map if it computes it

**Water bias and IDR:**

- `-water-bias` (flag): enable water bias
- `-water-bias-eps` (list of `str`): e.g., `H:3.6 C:2.1 idr:2.1`
- `-id-regions` (list of `str`): e.g., `A-10:45 60:80`
- `-idr-tune` (flag): deprecated passthrough to martinize2

**Protein description and edits:**

- `-noscfix` (flag), `-scfix` (flag), `-cys` (str)
- `-mutate` (list of `str`): e.g., `A-PHE45:ALA PHE30:ALA`
- `-modify` (list of `str`): e.g., `A-ASP45:ASP0 ASP:ASP0 +HSE`
- `-nter`, `-cter`, `-nt`: terminus patches

**Diagnostics:**

- `-write-graph`, `-write-repair`, `-write-canon`
- `-v` (repeatable): increase martinize2 verbosity
- `-maxwarn` (list of `int`)

**Input selection, custom force fields and passthrough:**

- `-ignore` (list of `str`): residue names martinize2 ignores, e.g. `HOH LIG`
- `-model` (int): MODEL to read from multi-model PDB frames
- `-ff` (str, `"martini3001"`), `-ff-dir`, `-map-dir` (lists): target force field and extra force field / mapping directories for martinize2 (e.g. custom ligand parameters)
- `-posres-fc` (float): position restraint force constant (kJ/mol/nm^2), martinize2 `-pf`
- `-min-seq-sep` (int, `4`): minimum residue separation for intra-chain contacts in the frequency analysis. Contacts between different chains are never filtered by sequence separation. Not equivalent to `-go-res-dist`, which is a graph distance applied by martinize2
- `-martinize-extra` (str): any other martinize2 flags as one string, using the `=` form, e.g. `-martinize-extra="-bonds-fudge 1.4"`

**Reference frame:**

- `-force-frame` (int or `None`): use a specific `frame_####` instead of auto-selection

**Appending missing contacts:**

- `-add-missing` (flag): append high-frequency pairs from `missing_high_freq.itp` into the final `go_nbparams.itp`. The script includes the `[ nonbond_params ]` header on append, and no extra blank line is added. Distances for these pairs are average CA–CA over all frames (in nm), and only pairs with average ≤ `-go-up` are kept.

Run `python contact_freq.py -h` to view all flags with their default values.

---

## How distances are computed (nm)

When building `missing_high_freq.itp`, the script scans all frames and measures CA–CA distances for each missing high-frequency contact.

- PDB frames: CA coordinates are read directly from the ATOM/HETATM records (first MODEL, first alternate location) in angstroms and converted to nm by dividing by 10. Frames are processed in parallel with `-cpus` workers.
- CIF frames: a lightweight mmCIF reader provides angstrom coordinates, then the script converts to nm by dividing by 10.

It then averages per-pair distances across all frames (nm) and keeps the pair only if `average_distance ≤ -go-up`.

The Lennard-Jones minimum used in the ITP is:

```ini
rmin = avg / 2^(1/6)
epsilon = -go-eps  # kJ/mol
```

---

## Output

- `filtered_*.txt`, `annotated_*.txt`: per-frame filtered and annotated contacts
- `normalized_<type>.txt`: contact frequencies across frames
- `high_<type>.txt`: contacts with frequency ≥ threshold
- `best_frame.txt` (plus a ties file if needed)
- `topol.top`, `<frame>_CG.pdb`
- `go_nbparams.itp` (filtered as requested)
- `go_nbparams_mock_<type>.itp` (mock for bookkeeping)
- `missing_high_freq.itp` (optional source for `-add-missing`)
- `high_counts_per_frame.txt`, `go_counts_per_frame.txt`

All intermediate `.txt`, `.map`, and input frames are moved to `output_files/`.
The final CG PDB stays in the run directory.

---

## Notes and tips

- For single proteins (monomers) use `-type intra`.
- For complexes, use `-type inter` or `-type both`:
  - `inter`: keeps all intra pairs plus only high-frequency inter pairs
  - `intra`: keeps all intra pairs plus only high-frequency intra pairs
  - `both`: keeps only high-frequency pairs (intra and inter)
- The script feeds `martinize2` with `-go <frame_X.map>` corresponding to the same frame used by `-f` (PDB or CIF).
- If you see an MDAnalysis warning about missing element information, it is harmless for distance calculations.

---

## Reproducibility

The script logs the exact command line into `run.log` each time you run it.

---

## Citation

If you use this repository, please cite:

1. Cofas-Vargas, L. F., Olivos-Ramirez, G. E., Chwastyk, M., Moreira, R. A., Baker, J. L., Marrink, S. J., & Poma, A. B. (2024). Nanomechanical footprint of SARS-CoV-2 variants in complex with a potent nanobody by molecular simulations. *Nanoscale, 16*(40), 18824–18834. https://doi.org/10.1039/D4NR02074J

2. Olivos-Ramirez, G. E., Cofas-Vargas, L. F., Marrink, S. J., & Poma, A. B. (2025). An optimized contact map for GōMartini 3 enabling conformational changes in protein assemblies [Preprint]. *bioRxiv*. https://doi.org/10.1101/2025.11.14.688437
