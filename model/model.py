import numpy as np

class McCannCarts:
    def __init__(self, N, valley_idx, Delta, gamma0, gamma1, gamma2=0.0, gamma3=0.0, gamma4=0.0, E0=0.0):
        """
        Initialize a system following McCann's model for ABC-stacked graphene.

        Parameters:
        -----------
        N : int
            Number of layers in the system
        valley_idx : int, 1 or -1
            Valley index
        Delta : float
            Gap = 2*Delta
        gamma0 : float
            In-layer hopping
        gamma1 : float
            Nearest-layer vertical hopping
        gamma2 : float, optional
            Next-nearest layer hopping
        gamma3 : float, optional
            Hopping breaking rotational symmetry (trigonal warping)
        gamma4 : float, optional
            Hopping breaking electron-hole symmetry
        E0 : float, optional
            Constant proportional to the identity.
        """

        # Store the parameters
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.valley_idx = valley_idx
        self.Delta = Delta
        self.N = N
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.E0 = E0

    def X_at_k(self, px, py):
        """
        Evaluate the coupling term at the given cartesian k-mesh.
        """
        pi = self.valley_idx * px + 1j * py
        pi_dag = self.valley_idx * px - 1j * py

        # Define velocities with a=1, hbar=1
        v = (np.sqrt(3) / 2) * self.gamma0
        v3 = (np.sqrt(3) / 2) * self.gamma3

        # Isotropic term
        X = ((v * pi) ** self.N) / ((-self.gamma1)**(self.N - 1))

        # Trigonal warping from gamma3 (v3)
        if self.gamma3 != 0:
            X += ((self.N - 1) / ((-self.gamma1)**(self.N - 2))) * v3 * pi_dag * (v * pi)**(self.N - 2)

        # Trigonal warping from gamma2 (next-nearest layer)
        if self.gamma2 != 0:
            X += ((self.N - 2) * self.gamma2 / (2 * (-self.gamma1)**(self.N - 3))) * (v * pi)**(self.N - 3)

        return X

    def get_h0_at_k(self, px, py):
        """
        Calculate the total diagonal term h0.
        """
        h0_total = self.E0

        if self.gamma4 != 0:
            v = (np.sqrt(3) / 2) * self.gamma0
            v4 = (np.sqrt(3) / 2) * self.gamma4
            p_mod_sq = px**2 + py**2
            h3c = (2 * v * v4 * p_mod_sq) / self.gamma1
            h0_total = h0_total - h3c

        return h0_total

    def get_energy(self, px, py):
        """
        Calculate the energy bands of the system analytically.
        Returns (E_plus, E_minus)
        """
        h0_total = self.get_h0_at_k(px, py)
        X = self.X_at_k(px, py)
        epsilon = np.sqrt(self.Delta ** 2 + np.abs(X) ** 2)

        return h0_total + epsilon, h0_total - epsilon

    def get_eigenstate_components(self, px, py, band_idx):
        """
        Calculate the components of the eigenstate for a given band.
        Returns (psi_x, psi_y)
        """
        X = self.X_at_k(px, py)
        epsilon = np.sqrt(self.Delta ** 2 + np.abs(X) ** 2)

        if self.Delta >= 0:
            if band_idx == 0:
                psi_x, psi_y = self.Delta + epsilon, np.conj(X)
            else:
                psi_x, psi_y = -X, self.Delta + epsilon
        else:
            if band_idx == 0:
                psi_x, psi_y = -X, self.Delta - epsilon
            else:
                psi_x, psi_y = self.Delta - epsilon, np.conj(X)

        # Normalize
        norm = np.sqrt(np.abs(psi_x)**2 + np.abs(psi_y)**2)
        psi_x /= norm
        psi_y /= norm

        return psi_x, psi_y

    def derivate_h0_at_k(self, px, py, axis):
        if self.gamma4 == 0:
            return 0

        v = (np.sqrt(3) / 2) * self.gamma0
        v4 = (np.sqrt(3) / 2) * self.gamma4

        if axis == 0: # x
            return (4 * v * v4 * px) / self.gamma1
        elif axis == 1: # y
            return (4 * v * v4 * py) / self.gamma1
        else:
            print("Error: axis out of bounds")
            return

    def derivate_X_at_k(self, px, py, axis):
        pi = self.valley_idx * px + 1j * py
        pi_dag = self.valley_idx * px - 1j * py

        v = (np.sqrt(3) / 2) * self.gamma0
        v3 = (np.sqrt(3) / 2) * self.gamma3
        N = self.N

        # Isotropic term
        dX0_dpi = N * (v ** N ) * (pi / (-self.gamma1)) ** (N - 1)

        if axis == 0:
            dX = self.valley_idx * dX0_dpi
        elif axis == 1:
            dX = 1j * dX0_dpi

        # v3 term
        if self.gamma3 != 0:
            dX10_dpi = (((N - 1) * (N - 2)) / ((-self.gamma1) ** (N - 2))) * v ** (N - 2) * pi ** (N - 3) * v3 * pi_dag
            dX10_dpidag = ((N - 1) / ((-self.gamma1) ** (N - 2))) * (v * pi) ** (N - 2) * v3

            if axis == 0:
                dX += self.valley_idx * (dX10_dpi + dX10_dpidag)
            elif axis == 1:
                dX += 1j * (dX10_dpi - dX10_dpidag)

        # gamma2 term
        if self.gamma2 != 0:
            dX01_dpi = (((N - 2) * (N - 3) * self.gamma2) / (2 * (-self.gamma1) ** (N - 3))) * v ** (N - 3) * pi ** (N - 4)
            if axis == 0:
                dX += self.valley_idx * dX01_dpi
            elif axis == 1:
                dX += 1j * dX01_dpi

        return dX