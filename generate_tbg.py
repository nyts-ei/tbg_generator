import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

class TwistedBilayerGraphene:
    """
    Twisted Bilayer Graphene (TBG) Structure Generator
    """
    def __init__(self, n: int, m: int, a_cc: float = 1.42, d_interlayer: float = 3.35, c_vacuum: float = 20.0):
        self.n = n
        self.m = m
        self.a_cc = a_cc
        self.a_lat = a_cc * np.sqrt(3)
        self.d_interlayer = d_interlayer
        self.c_vacuum = c_vacuum
        
        self._calculate_supercell_parameters()
        self._generate_lattice()

    def _calculate_supercell_parameters(self):
        n, m = self.n, self.m
        g = math.gcd(n, m)
        n_p = n // g
        m_p = m // g

        N_sq = n_p**2 + n_p*m_p + m_p**2

        if (n_p - m_p) % 3 == 0:
            self.N_atoms_theory = 4 * N_sq // 3
            c11, c12 = (n_p - m_p) // 3, (n_p + 2*m_p) // 3
            c21, c22 = -(n_p + 2*m_p) // 3, (2*n_p + m_p) // 3
        else:
            self.N_atoms_theory = 4 * N_sq
            c11, c12 = n_p, m_p
            c21, c22 = -m_p, n_p + m_p

        cos_theta = (n_p**2 + 4*n_p*m_p + m_p**2) / (2 * N_sq)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        self.theta_rad = np.arccos(cos_theta)
        self.theta_deg = np.degrees(self.theta_rad)

        self.c11, self.c12 = c11, c12
        self.c21, self.c22 = c21, c22

    def _generate_lattice(self):
        a_lat = self.a_lat
        a1 = np.array([a_lat, 0])
        a2 = np.array([a_lat/2, a_lat*np.sqrt(3)/2])

        delta_A = np.array([0, 0])
        delta_B = (1/3)*a1 + (1/3)*a2

        R1 = self._rotation_matrix(self.theta_rad / 2)
        R2 = self._rotation_matrix(-self.theta_rad / 2)

        a1_L1 = np.dot(R1, a1)
        a2_L1 = np.dot(R1, a2)

        self.T1 = self.c11 * a1_L1 + self.c12 * a2_L1
        self.T2 = self.c21 * a1_L1 + self.c22 * a2_L1

        L_max = max(np.linalg.norm(self.T1), np.linalg.norm(self.T2), np.linalg.norm(self.T1+self.T2))
        N_cells = int(np.ceil(L_max / a_lat)) + 5

        i = np.arange(-N_cells, N_cells+1)
        j = np.arange(-N_cells, N_cells+1)
        I, J = np.meshgrid(i, j)
        R = I.flatten()[:, None] * a1 + J.flatten()[:, None] * a2
        
        coords_orig = np.vstack((R + delta_A, R + delta_B))
        self.coords_layer1 = np.dot(coords_orig, R1.T)
        self.coords_layer2 = np.dot(coords_orig, R2.T)

        mask_L1 = self._get_atoms_in_supercell(self.coords_layer1)
        mask_L2 = self._get_atoms_in_supercell(self.coords_layer2)

        layer1_pos_2d = self.coords_layer1[mask_L1]
        layer2_pos_2d = self.coords_layer2[mask_L2]

        self.count_L1 = len(layer1_pos_2d)
        self.count_L2 = len(layer2_pos_2d)
        self.total_count = self.count_L1 + self.count_L2
        
        self.mask_L1 = mask_L1
        self.mask_L2 = mask_L2

        layer1_pos = np.column_stack((layer1_pos_2d, np.full(self.count_L1, 0.0)))
        layer2_pos = np.column_stack((layer2_pos_2d, np.full(self.count_L2, self.d_interlayer)))

        layer1_pos = layer1_pos[np.lexsort((layer1_pos[:, 0], layer1_pos[:, 1]))]
        layer2_pos = layer2_pos[np.lexsort((layer2_pos[:, 0], layer2_pos[:, 1]))]

        self.all_positions = np.vstack([layer1_pos, layer2_pos])
        
        self.cell_corners = np.array([[0, 0], self.T1, self.T1+self.T2, self.T2, [0, 0]])

    @staticmethod
    def _rotation_matrix(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def _get_atoms_in_supercell(self, coords):
        M = np.column_stack((self.T1, self.T2))
        M_inv = np.linalg.inv(M)
        c = np.dot(coords, M_inv.T)
        c_rounded = np.round(c, 10)
        mask = (c_rounded[:, 0] >= 0) & (c_rounded[:, 0] < 1) & \
               (c_rounded[:, 1] >= 0) & (c_rounded[:, 1] < 1)
        return mask

    def print_info(self):
        print(f"--- Twisted Bilayer Graphene (n={self.n}, m={self.m}) ---")
        print(f"Twist angle: {self.theta_deg:.3f} degrees")
        print(f"Theoretical number of atoms in supercell: {self.N_atoms_theory} (per layer: {self.N_atoms_theory//2})")
        print(f"\nRotated supercell vectors:")
        print(f"  T1 = ({self.T1[0]:.6f}, {self.T1[1]:.6f}) Å")
        print(f"  T2 = ({self.T2[0]:.6f}, {self.T2[1]:.6f}) Å")
        print(f"\nGenerated number of atoms:")
        print(f"  Layer 1: {self.count_L1} atoms")
        print(f"  Layer 2: {self.count_L2} atoms")
        print(f"  Total: {self.total_count} atoms")
        if self.total_count == self.N_atoms_theory:
            print("=> SUCCESS: Count matches the theoretical formula!")
        else:
            print("=> FAILED: Count does not match the theoretical formula!!")
            
    def save_xyz(self, filepath: str):
        """Save atomic positions to an XYZ file."""
        with open(filepath, 'w') as f:
            f.write(f"{self.total_count}\n")
            f.write(f"TBG n={self.n} m={self.m} theta={self.theta_deg:.3f} T1=[{self.T1[0]:.6f}, {self.T1[1]:.6f}, 0.0] T2=[{self.T2[0]:.6f}, {self.T2[1]:.6f}, 0.0]\n")
            for pos in self.all_positions:
                f.write(f"C {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}\n")
        print(f"Saved structure in XYZ format: {filepath}")

    def plot_structure(self, filepath: str):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.coords_layer1[:, 0], self.coords_layer1[:, 1], c='blue', s=10, alpha=0.1)
        plt.scatter(self.coords_layer2[:, 0], self.coords_layer2[:, 1], c='red', s=10, alpha=0.1)

        plt.scatter(self.coords_layer1[self.mask_L1, 0], self.coords_layer1[self.mask_L1, 1], 
                    c='blue', s=40, alpha=0.9, label=f'Layer 1 (in cell): {self.count_L1}')
        plt.scatter(self.coords_layer2[self.mask_L2, 0], self.coords_layer2[self.mask_L2, 1], 
                    c='red', s=40, alpha=0.9, label=f'Layer 2 (in cell): {self.count_L2}')

        plt.plot(self.cell_corners[:, 0], self.cell_corners[:, 1], 'k-', linewidth=2, label='Supercell')

        margin = 2 * self.a_lat
        plt.xlim(np.min(self.cell_corners[:, 0]) - margin, np.max(self.cell_corners[:, 0]) + margin)
        plt.ylim(np.min(self.cell_corners[:, 1]) - margin, np.max(self.cell_corners[:, 1]) + margin)

        plt.gca().set_aspect('equal')
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.title(f'TBG Supercell (n={self.n}, m={self.m}, $\\theta$={self.theta_deg:.2f}$^\\circ$)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved structure visualization (PNG): {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Twisted Bilayer Graphene (TBG) Structure Generator")
    parser.add_argument("-n", type=int, required=True, help="Lattice vector index n")
    parser.add_argument("-m", type=int, required=True, help="Lattice vector index m")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--format", type=str, choices=['xyz', 'png', 'both'], default='both', help="Output format(s)")
    
    args = parser.parse_args()
    
    n, m = args.n, args.m
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    prefix = f"tbg_{n}_{m}"
    
    tbg = TwistedBilayerGraphene(n=n, m=m)
    tbg.print_info()
    
    if args.format in ['xyz', 'both']:
        xyz_path = os.path.join(out_dir, f"{prefix}_structure.xyz")
        tbg.save_xyz(xyz_path)
        
    if args.format in ['png', 'both']:
        png_path = os.path.join(out_dir, f"{prefix}_structure.png")
        tbg.plot_structure(png_path)

if __name__ == "__main__":
    main()
