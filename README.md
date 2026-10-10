
# Standalone Finite Element Solver (FEM)

A standalone linear-static Finite Element Analysis (FEA) solver built from scratch in Python—no commercial FEA software packages required.

The solver reads ABAQUS mesh/input files (`.inp`), solves 2D and 3D continuum mechanics problems, and generates post-processing `.vtk` files for visualization in ParaView or any open-source visualizer. Cross-platform compatible (runs on both Windows and Linux).

---

## Features

- **Mesh Import:** Native parsing of ABAQUS `.inp` input files (*NODE, *ELEMENT, *NSET, *ELSET, *SURFACE, *BOUNDARY, *CLOAD, *DLOAD/*DSLOAD).
- **Element Library:**
  - **2D Elements:** 3-node triangular (`CPS3`/`CPE3`), 4-node quadrilateral (`CPS4`/`CPE4`).
  - **3D Solid Elements:** 4-node tetrahedral (`C3D4`), 8-node hexahedral (`C3D8`).
- **Physics & Formulations:** Linear elasticity supporting Plane Stress, Plane Strain, and 3D Solid elasticity formulations.
- **Loads & Boundary Conditions:** Concentrated point forces, face/edge pressure loads, and prescribed nodal displacement constraints.
- **Results & Post-Processing:** Direct export to ASCII `.vtk` unstructured grid files containing nodal/elemental displacement, Cauchy strain/stress tensors, principal stresses, and von Mises equivalent stresses.

---

## Requirements

- Python 3.8 or higher
- `numpy`
- `scipy` (optional, recommended for accelerated sparse matrix solves)

Install dependencies via `pip`:
```bash
pip install numpy scipy
