# Twisted Bilayer Graphene (TBG) Structure Generator

This repository provides Python tool for generating commensurate Twisted Bilayer Graphene (TBG) supercell structures. 

It allows you to easily generate atomic coordinates for any twist angle defined by $(n, m)$ indices.

## Features
- **Commensurability Support**: Automatically handles the lattice symmetry conditions (e.g., $(n-m) \mod 3$ conditions).
- **Multi-format Output**:
  - Exports atomic coordinates in standard `.xyz` format for DFT or tight-binding calculations.
  - Generates 2D moiré pattern visualizations in `.png`.

## Theoretical Background
For a pair of commensurate indices $(n, m)$, the twist angle $\theta$ is determined by:

$$\cos \theta = \frac{n^2 + 4nm + m^2}{2(n^2 + nm + m^2)}$$

The number of atoms in the supercell $N$ is calculated based on the divisibility of $(n-m)$ by 3, ensuring the smallest possible primitive supercell is generated.

## Requirements
- Python 3.8+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Usage
Specify the lattice indices $n$ and $m$ via command-line arguments:

```bash
python generate_tbg.py -n 8 -m 3 --out_dir ./output --format both
```

### Arguments
- `-n`: Lattice vector index $n$ (required).
- `-m`: Lattice vector index $m$ (required).
- `--out_dir`: Directory to save the output files (default: `.`).
- `--format`: Output format. Options: `xyz`, `png`, or `both` (default: `both`).

## Example Output
- `tbg_n_m_structure.xyz`: Cartesian coordinates (Å) of all atoms in the bilayer.
- `tbg_n_m_structure.png`: scatter plot of the moiré supercell and atomic positions.

---

