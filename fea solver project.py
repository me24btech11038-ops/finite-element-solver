#!/usr/bin/env python3
# Finite Element Method (FEM) Solver - Linear Static
# Supports 2D (T3, Q4) and 3D (T4, H8) elements.
# Inputs: Abaqus .inp files | Outputs: VTK format for ParaView

import math
import os
import sys
import numpy as np

try:
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Element definitions: (dimension, node_count, family, formulation)
ELEMENT_TYPES = {
    "CPS3": (2, 3, "T3", "plane_stress"),
    "CPE3": (2, 3, "T3", "plane_strain"),
    "CPS4": (2, 4, "Q4", "plane_stress"),
    "CPS4R": (2, 4, "Q4", "plane_stress"),
    "CPE4": (2, 4, "Q4", "plane_strain"),
    "CPE4R": (2, 4, "Q4", "plane_strain"),
    "C3D4": (3, 4, "T4", "solid"),
    "C3D8": (3, 8, "H8", "solid"),
    "C3D8R": (3, 8, "H8", "solid"),
}

# Local node mappings for element faces
FACE_NODES = {
    "T3": {"S1": [0, 1], "S2": [1, 2], "S3": [2, 0]},
    "Q4": {"S1": [0, 1], "S2": [1, 2], "S3": [2, 3], "S4": [3, 0]},
    "T4": {"S1": [0, 1, 2], "S2": [0, 3, 1], "S3": [1, 3, 2], "S4": [2, 3, 0]},
    "H8": {
        "S1": [0, 1, 2, 3], "S2": [4, 5, 6, 7],
        "S3": [0, 1, 5, 4], "S4": [1, 2, 6, 5],
        "S5": [2, 3, 7, 6], "S6": [3, 0, 4, 7]
    }
}

# Reference natural coordinates
NAT_Q4 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
NAT_H8 = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=float)

G2 = 1.0 / math.sqrt(3.0)
GAUSS_2 = (-G2, G2)


class FEModel:
    def __init__(self):
        self.nodes = {}
        self.elements = {}
        self.nsets = {}
        self.elsets = {}
        self.surfaces = {}
        self.materials = {}
        self.sections = []
        self.boundaries = []
        self.cloads = []
        self.dloads = []
        self.ndim = 2

    def get_nodes_from_target(self, target):
        t = str(target).strip().upper()
        if t in self.nsets:
            return set(self.nsets[t])
        try:
            return {int(t)}
        except ValueError:
            return set()

    def get_elems_from_target(self, target):
        t = str(target).strip().upper()
        if t in self.elsets:
            return set(self.elsets[t])
        try:
            return {int(t)}
        except ValueError:
            return set()


def parse_abaqus_inp(filename):
    model = FEModel()
    
    with open(filename, 'r', errors='ignore') as f:
        raw_lines = f.readlines()

    lines = []
    for line in raw_lines:
        s = line.split("**")[0].strip()
        if not s:
            continue
        if lines and lines[-1].endswith(",") and not s.startswith("*"):
            lines[-1] += " " + s
        else:
            lines.append(s)

    i = 0
    n = len(lines)
    active_mat = None
    has_3d = False

    while i < n:
        line = lines[i]
        if not line.startswith("*"):
            i += 1
            continue

        parts = [p.strip() for p in line.split(",")]
        key = parts[0].lower()
        opts = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                opts[k.strip().lower()] = v.strip()
            else:
                opts[p.strip().lower()] = "TRUE"

        i += 1
        data_block = []
        while i < n and not lines[i].startswith("*"):
            data_block.append([t.strip() for t in lines[i].split(",") if t.strip()])
            i += 1

        if key == "*node":
            for row in data_block:
                nid = int(row[0])
                coords = [float(x) for x in row[1:]]
                model.nodes[nid] = np.array(coords, dtype=float)

        elif key == "*element":
            el_type = opts.get("type", "CPS4").upper()
            if el_type not in ELEMENT_TYPES:
                raise ValueError(f"Element type {el_type} not implemented.")
            if ELEMENT_TYPES[el_type][0] == 3:
                has_3d = True
            
            eids = set()
            for row in data_block:
                eid = int(row[0])
                conn = [int(x) for x in row[1:]]
                model.elements[eid] = (el_type, conn)
                eids.add(eid)
            
            if "elset" in opts:
                el_name = opts["elset"].upper()
                model.elsets.setdefault(el_name, set()).update(eids)

        elif key in ("*nset", "*elset"):
            is_nset = (key == "*nset")
            set_name = opts.get("nset" if is_nset else "elset", "ALL").upper()
            target_dict = model.nsets if is_nset else model.elsets
            gen = "generate" in opts

            ids = set()
            for row in data_block:
                if gen:
                    vals = [int(float(x)) for x in row if x]
                    step = vals[2] if len(vals) > 2 else 1
                    ids.update(range(vals[0], vals[1] + 1, step))
                else:
                    for item in row:
                        try:
                            ids.add(int(item))
                        except ValueError:
                            ref = item.upper()
                            if ref in target_dict:
                                ids.update(target_dict[ref])
            target_dict.setdefault(set_name, set()).update(ids)

        elif key == "*surface":
            surf_name = opts.get("name", "SURF").upper()
            surf_list = model.surfaces.setdefault(surf_name, [])
            for row in data_block:
                if len(row) >= 2:
                    face = row[1].upper()
                    for eid in model.get_elems_from_target(row[0]):
                        surf_list.append((eid, face))

        elif key == "*material":
            active_mat = opts.get("name", "MAT").upper()
            model.materials.setdefault(active_mat, {})

        elif key == "*elastic":
            if data_block:
                E_val = float(data_block[0][0])
                nu_val = float(data_block[0][1]) if len(data_block[0]) > 1 else 0.3
                mat_name = active_mat or "DEFAULT"
                model.materials.setdefault(mat_name, {})["E"] = E_val
                model.materials[mat_name]["nu"] = nu_val

        elif key in ("*solid section", "*shell section"):
            thk = 1.0
            if data_block and data_block[0]:
                try:
                    thk = float(data_block[0][0])
                except ValueError:
                    pass
            model.sections.append({
                "elset": opts.get("elset", "ALL").upper(),
                "material": opts.get("material", "DEFAULT").upper(),
                "thickness": thk
            })

        elif key == "*boundary":
            for row in data_block:
                if not row:
                    continue
                node_ref = row[0]
                if len(row) > 1 and row[1].upper() in ("ENCASTRE", "PINNED"):
                    bc_type = row[1].upper()
                    d1, d2 = (1, 6) if bc_type == "ENCASTRE" else (1, 3)
                    model.boundaries.append((node_ref, d1, d2, 0.0))
                elif len(row) >= 2:
                    d1 = int(float(row[1]))
                    d2 = int(float(row[2])) if len(row) > 2 else d1
                    val = float(row[3]) if len(row) > 3 else 0.0
                    model.boundaries.append((node_ref, d1, d2, val))

        elif key in ("*cload", "*cforce"):
            for row in data_block:
                if len(row) >= 3:
                    model.cloads.append((row[0], int(float(row[1])), float(row[2])))

        elif key in ("*dload", "*dsload"):
            for row in data_block:
                if len(row) >= 3:
                    model.dloads.append((row[0], row[1].upper(), float(row[2])))
                elif len(row) == 2:
                    model.dloads.append((row[0], "P", float(row[1])))

    if not model.nodes or not model.elements:
        raise ValueError("Invalid .inp file: missing nodes or elements.")
    if not model.materials:
        model.materials["DEFAULT"] = {"E": 210e9, "nu": 0.3}

    model.ndim = 3 if has_3d else 2
    return model


def constitutive_matrix(E, nu, mode):
    if mode == "plane_stress":
        c = E / (1.0 - nu**2)
        return c * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])
    elif mode == "plane_strain":
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]
        ])
    else:  # 3d solid
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c1 = 1.0 - nu
        c2 = (1.0 - 2.0 * nu) / 2.0
        return c * np.array([
            [c1, nu, nu, 0, 0, 0],
            [nu, c1, nu, 0, 0, 0],
            [nu, nu, c1, 0, 0, 0],
            [0, 0, 0, c2, 0, 0],
            [0, 0, 0, 0, c2, 0],
            [0, 0, 0, 0, 0, c2]
        ])


def shape_derivatives_q4(xi, eta):
    dN = np.zeros((2, 4))
    for i, (a, b) in enumerate(NAT_Q4):
        dN[0, i] = 0.25 * a * (1 + b * eta)
        dN[1, i] = 0.25 * (1 + a * xi) * b
    return dN


def shape_derivatives_h8(xi, eta, zeta):
    dN = np.zeros((3, 8))
    pt = [xi, eta, zeta]
    for i, s in enumerate(NAT_H8):
        for k in range(3):
            val = s[k]
            for m in range(3):
                if m != k:
                    val *= (1 + pt[m] * s[m])
            dN[k, i] = 0.125 * val
    return dN


def get_B_matrix(family, coords, nat_coords=None):
    if family == "T3":
        (x1, y1), (x2, y2), (x3, y3) = coords[0][:2], coords[1][:2], coords[2][:2]
        detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * detJ
        if abs(area) < 1e-12:
            raise ValueError("Zero area in T3 element.")
        b = [y2 - y3, y3 - y1, y1 - y2]
        c = [x3 - x2, x1 - x3, x2 - x1]
        B = np.array([
            [b[0], 0, b[1], 0, b[2], 0],
            [0, c[0], 0, c[1], 0, c[2]],
            [c[0], b[0], c[1], b[1], c[2], b[2]]
        ]) / (2.0 * area)
        return B, abs(area)

    elif family == "T4":
        P = np.ones((4, 4))
        P[:, 1:] = coords[:, :3]
        vol = abs(np.linalg.det(P)) / 6.0
        if vol < 1e-12:
            raise ValueError("Zero volume in T4 element.")
        dN_dx = np.linalg.inv(P)[1:, :]
        B = np.zeros((6, 12))
        for i in range(4):
            gx, gy, gz = dN_dx[:, i]
            B[0, 3*i] = gx
            B[1, 3*i+1] = gy
            B[2, 3*i+2] = gz
            B[3, 3*i] = gy;   B[3, 3*i+1] = gx
            B[4, 3*i+1] = gz; B[4, 3*i+2] = gy
            B[5, 3*i] = gz;   B[5, 3*i+2] = gx
        return B, vol

    elif family == "Q4":
        xi, eta = nat_coords
        dN = shape_derivatives_q4(xi, eta)
        J = dN @ coords[:, :2]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Bad element Jacobian in Q4.")
        dN_dx = np.linalg.solve(J, dN)
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dN_dx[0, i]
            B[1, 2*i+1] = dN_dx[1, i]
            B[2, 2*i] = dN_dx[1, i]
            B[2, 2*i+1] = dN_dx[0, i]
        return B, detJ

    elif family == "H8":
        xi, eta, zeta = nat_coords
        dN = shape_derivatives_h8(xi, eta, zeta)
        J = dN @ coords[:, :3]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Bad element Jacobian in H8.")
        dN_dx = np.linalg.solve(J, dN)
        B = np.zeros((6, 24))
        for i in range(8):
            gx, gy, gz = dN_dx[:, i]
            B[0, 3*i] = gx
            B[1, 3*i+1] = gy
            B[2, 3*i+2] = gz
            B[3, 3*i] = gy;   B[3, 3*i+1] = gx
            B[4, 3*i+1] = gz; B[4, 3*i+2] = gy
            B[5, 3*i] = gz;   B[5, 3*i+2] = gx
        return B, detJ


def element_stiffness(family, coords, D, thickness):
    if family == "T3":
        B, area = get_B_matrix("T3", coords)
        return (B.T @ D @ B) * area * thickness
    elif family == "T4":
        B, vol = get_B_matrix("T4", coords)
        return (B.T @ D @ B) * vol

    ndofs = 8 if family == "Q4" else 24
    K_elem = np.zeros((ndofs, ndofs))
    
    if family == "Q4":
        for xi in GAUSS_2:
            for eta in GAUSS_2:
                B, detJ = get_B_matrix("Q4", coords, (xi, eta))
                K_elem += (B.T @ D @ B) * detJ * thickness
    elif family == "H8":
        for xi in GAUSS_2:
            for eta in GAUSS_2:
                for ze in GAUSS_2:
                    B, detJ = get_B_matrix("H8", coords, (xi, eta, ze))
                    K_elem += (B.T @ D @ B) * detJ
    return K_elem


def calculate_von_mises(s):
    if len(s) == 3:
        sxx, syy, sxy = s
        szz = 0.0
        syz, szx = 0.0, 0.0
    else:
        sxx, syy, szz, sxy, syz, szx = s
    return math.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) + 3.0 * (sxy**2 + syz**2 + szx**2))


def solve_fem(inp_path, vtk_path):
    print(f"Reading input file: {inp_path}")
    model = parse_abaqus_inp(inp_path)

    node_ids = sorted(model.nodes.keys())
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    ndim = model.ndim
    num_nodes = len(node_ids)
    num_dofs = num_nodes * ndim

    # Default section mapping
    elem_mat = {}
    elem_thk = {}
    default_mat = next(iter(model.materials))
    for eid in model.elements:
        elem_mat[eid] = default_mat
        elem_thk[eid] = 1.0

    for sec in model.sections:
        e_targets = model.get_elems_from_target(sec["elset"])
        for eid in e_targets:
            if eid in model.elements:
                elem_mat[eid] = sec["material"]
                elem_thk[eid] = sec["thickness"]

    print(f"Nodes: {num_nodes} | Elements: {len(model.elements)} | DOFs: {num_dofs}")

    # Assembly
    rows, cols, vals = [], [], []
    elem_cache = {}

    for eid, (el_type, conn) in model.elements.items():
        _, _, family, mode = ELEMENT_TYPES[el_type]
        coords = np.array([model.nodes[nid] for nid in conn], dtype=float)
        mat_info = model.materials.get(elem_mat[eid], model.materials[default_mat])
        D = constitutive_matrix(mat_info["E"], mat_info["nu"], mode)
        
        Ke = element_stiffness(family, coords, D, elem_thk[eid])
        edofs = np.array([ndim * node_map[nid] + d for nid in conn for d in range(ndim)])
        
        elem_cache[eid] = (family, mode, coords, D, edofs)
        rows.append(np.repeat(edofs, len(edofs)))
        cols.append(np.tile(edofs, len(edofs)))
        vals.append(Ke.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    if HAS_SCIPY:
        K_global = sps.coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()
    else:
        K_global = np.zeros((num_dofs, num_dofs))
        np.add.at(K_global, (rows, cols), vals)

    # Force vector
    F_global = np.zeros(num_dofs)
    for target, dof, mag in model.cloads:
        for nid in model.get_nodes_from_target(target):
            if nid in node_map and 1 <= dof <= ndim:
                F_global[ndim * node_map[nid] + dof - 1] += mag

    # Apply Boundary Conditions
    prescribed = {}
    for target, d1, d2, val in model.boundaries:
        for nid in model.get_nodes_from_target(target):
            if nid in node_map:
                for dof in range(d1, min(d2, ndim) + 1):
                    prescribed[ndim * node_map[nid] + dof - 1] = val

    fixed_dofs = np.array(sorted(prescribed.keys()), dtype=int)
    free_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)

    U = np.zeros(num_dofs)
    U[fixed_dofs] = [prescribed[d] for d in fixed_dofs]

    print("Solving system equations...")
    if HAS_SCIPY:
        K_ff = K_global[free_dofs][:, free_dofs].tocsc()
        rhs = F_global[free_dofs] - (K_global[free_dofs][:, fixed_dofs] @ U[fixed_dofs])
        U[free_dofs] = spla.spsolve(K_ff, rhs)
    else:
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        rhs = F_global[free_dofs] - K_global[np.ix_(free_dofs, fixed_dofs)] @ U[fixed_dofs]
        U[free_dofs] = np.linalg.solve(K_ff, rhs)

    disp = U.reshape(num_nodes, ndim)
    print(f"Max Displacement: {np.max(np.linalg.norm(disp, axis=1)):.6e}")

    # Write output VTK
    write_vtk(vtk_path, model, node_ids, node_map, ndim, disp, elem_cache, U)
    print(f"Results written to: {vtk_path}")


def write_vtk(filename, model, node_ids, node_map, ndim, disp, elem_cache, U):
    eids = list(model.elements.keys())
    vtk_cell_types = {"T3": 5, "Q4": 9, "T4": 10, "H8": 12}

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nFEA Results\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(node_ids)} double\n")
        for nid in node_ids:
            pt = model.nodes[nid]
            z = pt[2] if len(pt) > 2 else 0.0
            f.write(f"{pt[0]:.6e} {pt[1]:.6e} {z:.6e}\n")

        total_entries = sum(1 + len(model.elements[e][1]) for e in eids)
        f.write(f"\nCELLS {len(eids)} {total_entries}\n")
        for eid in eids:
            conn = model.elements[eid][1]
            f.write(f"{len(conn)} " + " ".join(str(node_map[n]) for n in conn) + "\n")

        f.write(f"\nCELL_TYPES {len(eids)}\n")
        for eid in eids:
            family = ELEMENT_TYPES[model.elements[eid][0]][2]
            f.write(f"{vtk_cell_types[family]}\n")

        # Nodal displacements
        f.write(f"\nPOINT_DATA {len(node_ids)}\n")
        f.write("VECTORS Displacement double\n")
        for i in range(len(node_ids)):
            u = disp[i]
            uz = u[2] if ndim == 3 else 0.0
            f.write(f"{u[0]:.6e} {u[1]:.6e} {uz:.6e}\n")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        inp_file = sys.argv[1]
        vtk_file = sys.argv[2] if len(sys.argv) >= 3 else os.path.splitext(inp_file)[0] + ".vtk"
        solve_fem(inp_file, vtk_file)
    else:
        print("Usage: python fem_solver.py <model.inp> [output.vtk]")