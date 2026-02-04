from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import numpy as np

from ..data.diagnostic import Diagnostic
from .postprocess import PostProcess

__all__ = ["Derivative_Diagnostic", "Derivative_Simulation", "Derivative_Species_Handler"]


class Derivative_Simulation(PostProcess):
    """Class to compute the derivative of a diagnostic. Works as a wrapper for the Derivative_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    This class can be initialized with either a Simulation object or another Derivative_Simulation object,
    allowing for chaining derivatives (e.g., second derivative = Derivative_Simulation(Derivative_Simulation(...))).

    Parameters
    ----------
    simulation : Simulation or Derivative_Simulation
        The simulation object or another derivative simulation object.
    deriv_type : str
        The type of derivative to compute. Options are:
        - 't' for time derivative.
        - 'x1' for first spatial derivative.
        - 'x2' for second spatial derivative.
        - 'x3' for third spatial derivative.
        - 'xx' for second spatial derivative in two axis.
        - 'xt' for mixed derivative in time and one spatial axis.
        - 'tx' for mixed derivative in one spatial axis and time.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types.
    order : int
        Scheme selector (2 or 4). Ignored if stencil is provided.
    stencil : Iterable[int] or None
        Integer offsets (in grid steps), e.g. [-2,-1,0,1,2] or [0,1,2,3].
    deriv_order : int
        Derivative order (1 for first derivative, 2 for second derivative, ...).
    """

    def __init__(
        self,
        simulation: Any,
        deriv_type: str,
        axis: int | tuple[int, int] | None = None,
        order: int = 4,
        stencil: Iterable[int] | None = None,
        deriv_order: int = 1,
    ):
        super().__init__(f"Derivative({deriv_type})", simulation)

        # Capability checks (supports chaining)
        if not hasattr(simulation, "__getitem__"):
            raise TypeError("simulation must be Simulation-like: supports __getitem__.")
        if not (hasattr(simulation, "_species") or hasattr(simulation, "species")):
            raise TypeError("simulation must be Simulation-like: has _species or species.")

        self._deriv_type = deriv_type
        self._op_axis = axis
        self._order = int(order)

        # NEW (general FD options)
        self._stencil = None if stencil is None else tuple(int(s) for s in stencil)
        self._deriv_order = int(deriv_order)

        self._derivatives_computed: dict[Any, Derivative_Diagnostic] = {}
        self._species_handler: dict[Any, Derivative_Species_Handler] = {}

        self._species = getattr(simulation, "_species", getattr(simulation, "species", []))

    def __getitem__(self, key: Any):
        if key in self._species:
            if key not in self._species_handler:
                self._species_handler[key] = Derivative_Species_Handler(
                    self._simulation[key],
                    self._deriv_type,
                    self._op_axis,
                    self._order,
                    stencil=self._stencil,
                    deriv_order=self._deriv_order,
                )
            return self._species_handler[key]

        if key not in self._derivatives_computed:
            self._derivatives_computed[key] = Derivative_Diagnostic(
                diagnostic=self._simulation[key],
                deriv_type=self._deriv_type,
                axis=self._op_axis,
                order=self._order,
                stencil=self._stencil,
                deriv_order=self._deriv_order,
            )
        return self._derivatives_computed[key]

    def delete_all(self) -> None:
        self._derivatives_computed = {}
        self._species_handler = {}

    def delete(self, key: Any) -> None:
        if key in self._derivatives_computed:
            del self._derivatives_computed[key]
        elif key in self._species_handler:
            del self._species_handler[key]
        else:
            print(f"Derivative {key} not found")

    @property
    def species(self) -> list:
        """Return list of species, making this compatible with Simulation interface"""
        return self._species

    @property
    def loaded_diagnostics(self) -> dict:
        """Return loaded diagnostics, making this compatible with Simulation interface"""
        return self._derivatives_computed

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """Add a custom diagnostic to the derivative simulation."""
        if name is None:
            i = 1
            while f"custom_diag_{i}" in self._derivatives_computed:
                i += 1
            name = f"custom_diag_{i}"

        if isinstance(diagnostic, Diagnostic):
            self._derivatives_computed[name] = diagnostic
            return name
        raise ValueError("Only Diagnostic objects are supported")


class Derivative_Diagnostic(Diagnostic):
    """Auxiliar class to compute the derivative of a diagnostic, for it to be similar in behavior to a Diagnostic object.
    Inherits directly from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    deriv_type : str
        The type of derivative to compute. Options are: 't', 'x1', 'x2', 'x3', 'xx', 'xt' and 'tx'.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types
    order : int
        Scheme selector (2 or 4). Ignored if stencil is provided.
    stencil : Iterable[int] or None
        Integer offsets (in grid steps), e.g. [-2,-1,0,1,2] or [0,1,2,3].
    deriv_order : int
        Derivative order (1 for first derivative, 2 for second derivative, ...).
        Only used if stencil is provided.

    Methods
    -------
    load_all()
        Load all the data and compute the derivative.
    __getitem__(index)
        Get data at a specific index.

    """

    def __init__(
        self,
        diagnostic: Diagnostic,
        deriv_type: str,
        axis: int | tuple[int, int] | None = None,
        order: int = 4,
        stencil: Iterable[int] | None = None,
        deriv_order: int = 1,
    ) -> None:
        # Initialize using parent's __init__ with the same species
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=(getattr(diagnostic, "_simulation_folder", None)),
                species=getattr(diagnostic, "_species", None),
            )
        else:
            super().__init__(None)

        self.postprocess_name = "DERIV"

        self._diag = diagnostic
        self._deriv_type = deriv_type
        self._op_axis = axis
        self._data = None
        self._all_loaded = False
        self._order = int(order)

        self._stencil = None if stencil is None else tuple(int(s) for s in stencil)
        self._deriv_order = int(deriv_order)

        self._cache = OrderedDict()
        self._cache_max = 6  # Maximum number of items to keep in cache (enough for 4th-order time stencil)

        # Copy all relevant attributes from diagnostic
        for attr in [
            "_dt",
            "_dx",
            "_ndump",
            "_axis",
            "_nx",
            "_x",
            "_grid",
            "_dim",
            "_maxiter",
            "_type",
        ]:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    @staticmethod
    def _validate_stencil(stencil: tuple[int, ...], deriv_order: int) -> None:
        if len(stencil) == 0:
            raise ValueError("Stencil cannot be empty")
        if len(set(stencil)) != len(stencil):
            raise ValueError(f"Stencil offsets must be distinct. Got {stencil}.")
        if deriv_order < 0:
            raise ValueError("deriv_order must be >= 0")
        if deriv_order >= len(stencil):
            raise ValueError(f"Need len(stencil) > deriv_order. Got len={len(stencil)} deriv_order={deriv_order}.")

    # Compute FD coefficients for given stencil and derivative order
    @staticmethod
    @lru_cache(maxsize=2048)
    def _fd_unit_coeffs(stencil: tuple[int, ...], deriv_order: int) -> np.ndarray:
        """Unit-spacing coefficients (h=1). Scale by 1/h^d when applying.

        Parameters
        ----------
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).
        deriv_order : int
            The derivative order.

        Returns
        -------
        np.ndarray
            The finite difference coefficients.

        Maths
        -----
        Computes sum_j c_j s_j^n = n! * delta_{n,d},   n = 0..N-1
        """
        Derivative_Diagnostic._validate_stencil(stencil, deriv_order)

        s = np.asarray(stencil, dtype=float)
        N = s.size

        # V[n,j] = s_j^n, n=0..N-1
        V = np.vander(s, N=N, increasing=True).T  # (N,N)
        rhs = np.zeros(N, dtype=float)
        rhs[deriv_order] = math.factorial(deriv_order)

        return np.linalg.solve(V, rhs)

    @staticmethod
    def _shift_stencil_to_fit(stencil: np.ndarray, i: int, n: int) -> np.ndarray:
        """Shift stencil by an integer k so i + (s_j + k) in [0,n-1] for all j.
        Prefer k=0 if feasible.

        Parameters
        ----------
        stencil : np.ndarray
            The stencil offsets (in grid steps).
        i : int
            The current index where the stencil will be applied.
        n : int
            The total number of points along the axis.

        Returns
        -------
        np.ndarray
            The shifted stencil.

        Examples
        --------
        >>> s = np.array([-2, -1, 0, 1, 2])
        >>> Derivative_Diagnostic._shift_stencil_to_fit(s, i=1, n=10)
        array([-1,  0,  1,  2,  3])

        >>> Derivative_Diagnostic._shift_stencil_to_fit(s, i=9, n=10)
        array([5, 6, 7, 8, 9])

        >>> Derivative_Diagnostic._shift_stencil_to_fit(s, i=5, n=10)
        array([-2, -1,  0,  1,  2])
        """
        s = stencil.astype(int)

        k_min = -(10**18)
        k_max = 10**18
        for sj in s:
            k_min = max(k_min, -i - sj)
            k_max = min(k_max, (n - 1) - i - sj)

        if k_min > k_max:
            raise ValueError(f"Stencil {stencil.tolist()} cannot be shifted to fit at i={i}, n={n}.")

        if k_min <= 0 <= k_max:
            k = 0
        else:
            # pick nearest feasible integer
            k = k_min if abs(k_min) < abs(k_max) else k_max

        return s + int(k)

    @staticmethod
    @lru_cache(maxsize=128)
    def _edge_plan(n: int, axis: int, stencil: tuple[int, ...], deriv_order: int):
        """Precompute which indices need shifted stencils and what those shifted stencils are.

        Parameters
        ----------
        n : int
            Number of points along the axis.
        axis : int
            Axis along which the derivative is computed.
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).
        deriv_order : int
            The derivative order.

        Returns
        -------
        idxs : list[int]
            Indices where shifted stencils are needed.
        stencils : list[tuple[int,...]]
            Shifted stencils (same length as idxs)

        Examples
        --------
        >>> Derivative_Diagnostic._edge_plan(n=10, axis=1, stencil=(-2, -1, 0, 1, 2), deriv_order=1)
        ([0, 1, 8, 9], [(-2, -1, 0, 1, 2), (-1, 0, 1, 2, 3), (5, 6, 7, 8, 9), (6, 7, 8, 9, 10)])
        """

        # Pass stencil as np array for easier min/max
        s0 = np.asarray(stencil, dtype=int)
        smin = int(s0.min())  # minimum shift
        smax = int(s0.max())  # maximum shift

        start = -smin
        stop = n - smax  # interior indices are [start, stop)
        idxs: list[int] = []
        stencils: list[tuple[int, ...]] = []

        # left edge
        for i in range(0, max(0, start)):
            s_i = Derivative_Diagnostic._shift_stencil_to_fit(s0, i=i, n=n)
            idxs.append(i)
            stencils.append(tuple(int(x) for x in s_i.tolist()))

        # right edge
        for i in range(max(0, stop), n):
            s_i = Derivative_Diagnostic._shift_stencil_to_fit(s0, i=i, n=n)
            idxs.append(i)
            stencils.append(tuple(int(x) for x in s_i.tolist()))

        return idxs, stencils

    def _apply_stencil_unshifted_interior(
        self,
        data: np.ndarray,
        h: float,
        axis: int,
        stencil: tuple[int, ...],
        deriv_order: int,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Apply unshifted stencil on the interior region where it fits. Returns (out, (start,stop)).

        Parameters
        ----------
        data : np.ndarray
            The input data array.
        h : float
            The grid spacing.
        axis : int
            The axis along which to apply the stencil.
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).
        deriv_order : int
            The derivative order.

        Returns
        -------
        out : np.ndarray
            The output array with the derivative applied in the interior region.
        (start, stop) : tuple[int, int]
            The start and stop indices of the interior region where the stencil was applied.
        """
        s0 = np.asarray(stencil, dtype=int)
        n = data.shape[axis]
        smin = int(s0.min())
        smax = int(s0.max())

        start = -smin
        stop = n - smax
        out = np.zeros_like(data, dtype=float)

        if stop <= start:
            return out, (start, stop)

        c_unit = self._fd_unit_coeffs(stencil, deriv_order)
        c = c_unit / (float(h) ** deriv_order)

        tgt = [slice(None)] * data.ndim
        tgt[axis] = slice(start, stop)
        interior_view = data[tuple(tgt)]

        acc = np.zeros_like(interior_view, dtype=float)
        for cj, sj in zip(c, s0, strict=False):
            src = [slice(None)] * data.ndim
            src[axis] = slice(start + sj, stop + sj)
            acc += cj * data[tuple(src)]
        out[tuple(tgt)] = acc

        return out, (start, stop)

    def _fd_apply_along_axis(
        self,
        data: np.ndarray,
        h: float,
        axis: int,
        deriv_order: int,
        stencil: tuple[int, ...],
    ) -> np.ndarray:
        """Apply arbitrary stencil along an axis:
        - vectorized interior (unshifted)
        - precomputed shifted stencils at edges

        Parameters
        ----------
        data : np.ndarray
            The input data array.
        h : float
            The grid spacing.
        axis : int
            The axis along which to apply the stencil.
        deriv_order : int
            The derivative order.
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).

        Returns
        -------
        np.ndarray
            The output array with the derivative applied.
        """
        self._validate_stencil(stencil, deriv_order)

        n = data.shape[axis]
        # Central interior (vectorized)
        out, (start, stop) = self._apply_stencil_unshifted_interior(data, h, axis, stencil, deriv_order)

        # edges plan + apply (small loops only over edge indices)
        idxs, stencils = self._edge_plan(n, axis, stencil, deriv_order)

        # Apply edge stencils
        for i, s_i in zip(idxs, stencils, strict=False):
            # Compute coefficients for this shifted stencil
            c_unit = self._fd_unit_coeffs(s_i, deriv_order)
            # Scale by h^d
            c_i = c_unit / (float(h) ** deriv_order)
            # Convert stencil to array of ints
            s_arr = np.asarray(s_i, dtype=int)

            # Target index
            tgt = [slice(None)] * data.ndim
            tgt[axis] = i

            # Apply stencil at index i
            acc = 0.0
            for cj, sj in zip(c_i, s_arr, strict=False):
                src = [slice(None)] * data.ndim
                src[axis] = i + int(sj)
                acc = acc + cj * data[tuple(src)]
            out[tuple(tgt)] = acc

        return out

    @staticmethod
    def _compute_fourth_order_spatial(data: np.ndarray, dx: float, axis: int) -> np.ndarray:
        """Compute 4th-order spatial derivative along specified axis using vectorized operations.

        Uses the 4th-order central difference stencil:
        Interior: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)
        Boundaries: 2nd-order forward/backward differences

        Parameters
        ----------
        data : np.ndarray
            Input data array
        dx : float
            Grid spacing
        axis : int
            Axis along which to compute derivative

        Returns
        -------
        np.ndarray
            Derivative of the input data

        """
        result = np.zeros_like(data)

        # Build slice objects for vectorized operations
        # This is much faster than looping through indices

        # Central differences (vectorized)
        # For each point i in [2, n-2), compute:
        # (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)

        slices_center = [slice(None)] * data.ndim
        slices_p2 = [slice(None)] * data.ndim
        slices_p1 = [slice(None)] * data.ndim
        slices_m1 = [slice(None)] * data.ndim
        slices_m2 = [slice(None)] * data.ndim

        # Target region: indices 2 to -2
        slices_center[axis] = slice(2, -2)
        # For the stencil, we need aligned slices
        slices_p2[axis] = slice(4, None)  # i+2: starts at 4, goes to end
        slices_p1[axis] = slice(3, -1)  # i+1: starts at 3, goes to -1
        slices_m1[axis] = slice(1, -3)  # i-1: starts at 1, goes to -3
        slices_m2[axis] = slice(0, -4)  # i-2: starts at 0, goes to -4

        result[tuple(slices_center)] = (
            -data[tuple(slices_p2)] + 8 * data[tuple(slices_p1)] - 8 * data[tuple(slices_m1)] + data[tuple(slices_m2)]
        ) / (12 * dx)

        # Boundary points using 2nd-order differences
        # First point: forward difference
        slices_0 = [slice(None)] * data.ndim
        slices_1 = [slice(None)] * data.ndim
        slices_2 = [slice(None)] * data.ndim
        slices_0[axis] = 0
        slices_1[axis] = 1
        slices_2[axis] = 2
        result[tuple(slices_0)] = (-3 * data[tuple(slices_0)] + 4 * data[tuple(slices_1)] - data[tuple(slices_2)]) / (2 * dx)

        # Second point: central difference
        result[tuple(slices_1)] = (data[tuple(slices_2)] - data[tuple(slices_0)]) / (2 * dx)

        # Second-to-last point: central difference
        slices_m2 = [slice(None)] * data.ndim
        slices_m1 = [slice(None)] * data.ndim
        slices_m3 = [slice(None)] * data.ndim
        slices_m2[axis] = -2
        slices_m1[axis] = -1
        slices_m3[axis] = -3
        result[tuple(slices_m2)] = (data[tuple(slices_m1)] - data[tuple(slices_m3)]) / (2 * dx)

        # Last point: backward difference
        result[tuple(slices_m1)] = (3 * data[tuple(slices_m1)] - 4 * data[tuple(slices_m2)] + data[tuple(slices_m3)]) / (2 * dx)

        return result

    def load_all(self) -> np.ndarray:
        """Load all data and compute the derivative."""
        if self._data is not None:
            print("Using cached derivative")
            return self._data

        if not self._diag._all_loaded:
            self._diag.load_all()

        self._data = self._diag._data

        def dx_for_axis_data(ax: int) -> float:
            # data in load_all includes time axis at 0, spatial axes start at 1
            if self._dim > 1:
                return float(self._diag._dx[ax - 1])
            dx0 = self._diag._dx
            return float(dx0[0] if isinstance(dx0, (list, tuple, np.ndarray)) else dx0)

        if self._stencil is not None:
            self._validate_stencil(self._stencil, self._deriv_order)

            if self._deriv_type == "t":
                h = float(self._diag._dt * self._diag._ndump)
                result = self._fd_apply_along_axis(self._data, h=h, axis=0, deriv_order=self._deriv_order, stencil=self._stencil)

            elif self._deriv_type in ("x1", "x2", "x3"):
                axis_map = {"x1": 1, "x2": 2, "x3": 3}
                ax = axis_map[self._deriv_type]
                dx = dx_for_axis_data(ax)
                result = self._fd_apply_along_axis(self._data, h=dx, axis=ax, deriv_order=self._deriv_order, stencil=self._stencil)

            else:
                raise ValueError("Explicit stencil is supported for deriv_type in {'t','x1','x2','x3'} in load_all().")

            self._all_loaded = True
            self._data = result
            return self._data

        if self._order == 2:
            if self._deriv_type == "t":
                result = np.gradient(self._data, self._diag._dt * self._diag._ndump, axis=0, edge_order=2)

            elif self._deriv_type == "x1":
                result = np.gradient(self._data, dx_for_axis_data(1), axis=1, edge_order=2)

            elif self._deriv_type == "x2":
                result = np.gradient(self._data, dx_for_axis_data(2), axis=2, edge_order=2)

            elif self._deriv_type == "x3":
                result = np.gradient(self._data, dx_for_axis_data(3), axis=3, edge_order=2)

            elif self._deriv_type == "xx":
                if not isinstance(self._op_axis, (tuple, list)) or len(self._op_axis) != 2:
                    raise ValueError("Axis must be a tuple with two elements.")
                result = np.gradient(
                    np.gradient(
                        self._data,
                        self._diag._dx[self._op_axis[0] - 1],
                        axis=self._op_axis[0],
                        edge_order=2,
                    ),
                    self._diag._dx[self._op_axis[1] - 1],
                    axis=self._op_axis[1],
                    edge_order=2,
                )

            elif self._deriv_type == "tx":
                if not isinstance(self._op_axis, int):
                    raise ValueError("Axis must be an integer.")
                result = np.gradient(
                    np.gradient(
                        self._data,
                        self._diag._dx[self._op_axis - 1],
                        axis=self._op_axis,
                        edge_order=2,
                    ),
                    self._diag._dt,
                    axis=0,
                    edge_order=2,
                )
            else:
                raise ValueError("Invalid derivative type.")

        elif self._order == 4:
            if self._deriv_type in ("x1", "x2", "x3"):
                axis_map = {"x1": 1, "x2": 2, "x3": 3}
                ax = axis_map[self._deriv_type]
                dx = dx_for_axis_data(ax)
                result = self._compute_fourth_order_spatial(self._data, dx, ax)
            else:
                raise ValueError("Order 4 is only implemented for spatial derivatives 'x1', 'x2' and 'x3'.")
        else:
            raise ValueError("Only order 2 and 4 supported.")

        self._all_loaded = True
        self._data = result
        return self._data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        # dimension guards
        if self._deriv_type in ("x2", "xt", "tx") and self._diag._dim < 2:
            raise ValueError(f"{self._deriv_type} requested but diagnostic dim={self._diag._dim}")
        if self._deriv_type in ("x3",) and self._diag._dim < 3:
            raise ValueError(f"{self._deriv_type} requested but diagnostic dim={self._diag._dim}")
        if self._deriv_type == "xx":
            if not isinstance(self._op_axis, (tuple, list)) or len(self._op_axis) != 2:
                raise ValueError("For 'xx', axis must be a tuple/list of two spatial axes (1..3).")
            if max(self._op_axis) > self._diag._dim:
                raise ValueError(f"xx requested for axes {self._op_axis} but dim={self._diag._dim}")

        n = int(self._diag._maxiter)
        dt = float(self._diag._dt * self._diag._ndump)

        # ---------- helpers ----------
        def spatial_axis_np_from_osiris(ax_osiris: int) -> int:
            # OSIRIS spatial axes are 1..3; per-timestep frame axes are 0..2
            if not isinstance(ax_osiris, int):
                raise ValueError("Axis must be int for spatial derivatives.")
            if ax_osiris < 1 or ax_osiris > 3:
                raise ValueError(f"Spatial axis must be 1..3, got {ax_osiris}")
            return ax_osiris - 1

        def dx_for_np_axis(ax_np: int) -> float:
            dx = self._diag._dx
            if self._diag._dim == 1:
                return float(dx[0] if isinstance(dx, (list, tuple, np.ndarray)) else dx)
            return float(dx[ax_np])

        def d_dx_frame(f: np.ndarray, ax_np: int) -> np.ndarray:
            dx = dx_for_np_axis(ax_np)

            # stencil overrides legacy order
            if self._stencil is not None:
                self._validate_stencil(self._stencil, self._deriv_order)
                return self._fd_apply_along_axis(f, h=dx, axis=ax_np, deriv_order=self._deriv_order, stencil=self._stencil)

            if self._order == 2:
                return np.gradient(f, dx, axis=ax_np, edge_order=2)
            if self._order == 4:
                return self._compute_fourth_order_spatial(f, dx, ax_np)
            raise ValueError("Only order 2 and 4 supported.")

        def d_dt_at(i: int) -> np.ndarray:
            # stencil overrides legacy order
            if self._stencil is not None:
                self._validate_stencil(self._stencil, self._deriv_order)

                s0 = np.asarray(self._stencil, dtype=int)
                s_i = self._shift_stencil_to_fit(s0, i=i, n=n)
                s_i_tup = tuple(int(x) for x in s_i.tolist())

                c_unit = self._fd_unit_coeffs(s_i_tup, self._deriv_order)
                c = c_unit / (dt**self._deriv_order)

                acc = 0.0
                for cj, sj in zip(c, s_i, strict=False):
                    acc = acc + cj * self._base(i + int(sj), data_slice)
                return acc

            if self._order == 2:
                if i == 0:
                    f0 = self._base(0, data_slice)
                    f1 = self._base(1, data_slice)
                    f2 = self._base(2, data_slice)
                    return (-3 * f0 + 4 * f1 - f2) / (2 * dt)
                if i == n - 1:
                    f0 = self._base(n - 1, data_slice)
                    f1 = self._base(n - 2, data_slice)
                    f2 = self._base(n - 3, data_slice)
                    return (3 * f0 - 4 * f1 + f2) / (2 * dt)
                fp = self._base(i + 1, data_slice)
                fm = self._base(i - 1, data_slice)
                return (fp - fm) / (2 * dt)

            if self._order == 4:
                if i < 2 or i > n - 3:
                    # edge fallback to 2nd-order
                    if i == 0:
                        f0 = self._base(0, data_slice)
                        f1 = self._base(1, data_slice)
                        f2 = self._base(2, data_slice)
                        return (-3 * f0 + 4 * f1 - f2) / (2 * dt)
                    if i == n - 1:
                        f0 = self._base(n - 1, data_slice)
                        f1 = self._base(n - 2, data_slice)
                        f2 = self._base(n - 3, data_slice)
                        return (3 * f0 - 4 * f1 + f2) / (2 * dt)
                    fp = self._base(min(i + 1, n - 1), data_slice)
                    fm = self._base(max(i - 1, 0), data_slice)
                    return (fp - fm) / (2 * dt)

                f_p2 = self._base(i + 2, data_slice)
                f_p1 = self._base(i + 1, data_slice)
                f_m1 = self._base(i - 1, data_slice)
                f_m2 = self._base(i - 2, data_slice)
                return (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * dt)

            raise ValueError("Only order 2 and 4 supported.")

        if self._deriv_type in ("x1", "x2", "x3"):
            f = self._base(index, data_slice)
            ax_np = {"x1": 0, "x2": 1, "x3": 2}[self._deriv_type]
            return d_dx_frame(f, ax_np)

        if self._deriv_type == "t":
            return d_dt_at(index)

        if self._deriv_type == "xx":
            ax1_np = spatial_axis_np_from_osiris(self._op_axis[0])
            ax2_np = spatial_axis_np_from_osiris(self._op_axis[1])

            f = self._base(index, data_slice)

            g = np.gradient(f, dx_for_np_axis(ax1_np), axis=ax1_np, edge_order=2)
            return np.gradient(g, dx_for_np_axis(ax2_np), axis=ax2_np, edge_order=2)

        if self._deriv_type == "xt":
            if not isinstance(self._op_axis, int):
                raise ValueError("For 'xt', axis must be an OSIRIS spatial axis int (1..3).")
            ax_np = spatial_axis_np_from_osiris(self._op_axis)

            ft = d_dt_at(index)
            return np.gradient(ft, dx_for_np_axis(ax_np), axis=ax_np, edge_order=2)

        if self._deriv_type == "tx":
            if not isinstance(self._op_axis, int):
                raise ValueError("For 'tx', axis must be an OSIRIS spatial axis int (1..3).")
            ax_np = spatial_axis_np_from_osiris(self._op_axis)

            def spatial_frame(i: int) -> np.ndarray:
                fi = self._base(i, data_slice)
                return np.gradient(fi, dx_for_np_axis(ax_np), axis=ax_np, edge_order=2)

            if self._order == 2:
                if index == 0:
                    return (-3 * spatial_frame(0) + 4 * spatial_frame(1) - spatial_frame(2)) / (2 * dt)
                if index == n - 1:
                    return (3 * spatial_frame(n - 1) - 4 * spatial_frame(n - 2) + spatial_frame(n - 3)) / (2 * dt)
                return (spatial_frame(index + 1) - spatial_frame(index - 1)) / (2 * dt)

            if self._order == 4:
                if index < 2 or index > n - 3:
                    return (spatial_frame(min(index + 1, n - 1)) - spatial_frame(max(index - 1, 0))) / (2 * dt)
                return (
                    -spatial_frame(index + 2) + 8 * spatial_frame(index + 1) - 8 * spatial_frame(index - 1) + spatial_frame(index - 2)
                ) / (12 * dt)

            raise ValueError("Only order 2 and 4 supported.")

        raise ValueError("Invalid derivative type.")

    def _cache_get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key, val):
        self._cache[key] = val
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _base(self, idx: int, data_slice: tuple | None):
        key = (idx, data_slice)
        v = self._cache_get(key)
        if v is None:
            v = self._diag._frame(idx, data_slice=data_slice)
            self._cache_put(key, v)
        return v


class Derivative_Species_Handler:
    """Class to handle derivatives for a species.
    Acts as a wrapper for the Derivative_Diagnostic class.

    Not intended to be used directly, but through the Derivative_Simulation class.

    Parameters
    ----------
    species_handler : Species_Handler
        The species handler object.
    type : str
        The type of derivative to compute. Options are: 't', 'x1', 'x2', 'x3', 'xx', 'xt' and 'tx'.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types.

    """

    def __init__(
        self,
        species_handler: Any,
        deriv_type: str,
        axis: int | tuple[int, int] | None = None,
        order: int = 4,
        stencil: Iterable[int] | None = None,
        deriv_order: int = 1,
    ) -> None:
        self._species_handler = species_handler
        self._deriv_type = deriv_type
        self._axis = axis
        self._order = int(order)

        self._stencil = None if stencil is None else tuple(int(s) for s in stencil)
        self._deriv_order = int(deriv_order)

        self._derivatives_computed: dict[Any, Derivative_Diagnostic] = {}

    def __getitem__(self, key: Any) -> Derivative_Diagnostic:
        if key not in self._derivatives_computed:
            diag = self._species_handler[key]
            self._derivatives_computed[key] = Derivative_Diagnostic(
                diagnostic=diag,
                deriv_type=self._deriv_type,
                axis=self._axis,
                order=self._order,
                stencil=self._stencil,
                deriv_order=self._deriv_order,
            )
        return self._derivatives_computed[key]
