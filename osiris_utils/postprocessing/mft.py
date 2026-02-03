from __future__ import annotations

import numpy as np

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess

__all__ = [
    "MFT_Simulation",
    "MFT_Diagnostic",
    "MFT_Species_Handler",
]


class MFT_Simulation(PostProcess):
    """
    Class to compute the Mean Field Theory approximation of a diagnostic. Works as a wrapper for the MFT_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    mft_axis : int
        The axis to compute the mean field theory.

    """

    def __init__(self, simulation: Simulation, mft_axis: int | None = None):
        super().__init__(f"MeanFieldTheory({mft_axis})")
        if not isinstance(simulation, Simulation):
            raise ValueError("simulation must be a Simulation-compatible object.")
        self._simulation = simulation
        self._mft_axis = mft_axis
        self._mft_computed = {}
        self._species_handler = {}

    def __getitem__(self, key):
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = MFT_Species_Handler(self._simulation[key], self._mft_axis)
            return self._species_handler[key]
        if key not in self._mft_computed:
            self._mft_computed[key] = MFT_Diagnostic(self._simulation[key], self._mft_axis)
        return self._mft_computed[key]

    def delete_all(self):
        self._mft_computed = {}

    def delete(self, key):
        if key in self._mft_computed:
            del self._mft_computed[key]
        else:
            print(f"MeanFieldTheory {key} not found in simulation")

    def process(self, diagnostic):
        """Apply mean field theory to a diagnostic"""
        return MFT_Diagnostic(diagnostic, self._mft_axis)


class MFT_Diagnostic(Diagnostic):
    """
    Class to compute mean field theory of a diagnostic.
    Acts as a container for the average and fluctuation components.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    mft_axis : int
        The axis to compute mean field theory along.


    """

    def __init__(self, diagnostic, mft_axis):
        # Initialize using parent's __init__ with the same species
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=(diagnostic._simulation_folder if hasattr(diagnostic, "_simulation_folder") else None),
                species=diagnostic._species,
            )
        else:
            super().__init__(None)

        self._name = f"MFT[{diagnostic._name}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False

        # Components that will be lazily created
        self._components = {}

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
            "_tunits",
            "_type",
        ]:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    def __getitem__(self, key):
        """
        Get a component of the mean field theory.

        Parameters
        ----------
        key : str
            Either "avg" for average or "delta" for fluctuations.

        Returns
        -------
        Diagnostic
            The requested component.
        """
        if key == "avg":
            if "avg" not in self._components:
                self._components["avg"] = MFT_Diagnostic_Average(self._diag, self._mft_axis)
            return self._components["avg"]

        elif key == "delta":
            if "delta" not in self._components:
                self._components["delta"] = MFT_Diagnostic_Fluctuations(self._diag, self._mft_axis)
            return self._components["delta"]

        else:
            raise ValueError("Invalid MFT component. Use 'avg' or 'delta'.")

    def load_all(self):
        """Load both average and fluctuation components"""
        # This will compute both components at once for efficiency
        if "avg" not in self._components:
            self._components["avg"] = MFT_Diagnostic_Average(self._diag, self._mft_axis)

        if "delta" not in self._components:
            self._components["delta"] = MFT_Diagnostic_Fluctuations(self._diag, self._mft_axis)

        # Load both components
        self._components["avg"].load_all()
        self._components["delta"].load_all()

        # Mark this container as loaded
        self._all_loaded = True

        return self._components


class MFT_Diagnostic_Average(Diagnostic):
    """
    Class to compute the average component of mean field theory.
    Inherits from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    mft_axis : int
        The axis to compute the mean field theory.

    """

    def __init__(self, diagnostic, mft_axis):
        # Initialize with the same species as the diagnostic
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=(diagnostic._simulation_folder if hasattr(diagnostic, "_simulation_folder") else None),
                species=diagnostic._species,
            )
        else:
            super().__init__(None)

        if mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")

        self.postprocess_name = "MFT_AVG"

        self._name = f"MFT_avg[{diagnostic._name}, {mft_axis}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False

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

    def load_all(self):
        """Load all data and compute the average"""
        if self._diag._all_loaded is True:
            print("Diagnostic data already loaded ... applyting MFT")
            self._data = self._diag._data
        if self._data is not None:
            print("Data already loaded")
            return self._data

        if not hasattr(self._diag, "_data") or self._diag._data is None:
            self._diag.load_all()

        if self._mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")
        else:
            self._data = np.expand_dims(self._diag._data.mean(axis=self._mft_axis), axis=-1)

        self._all_loaded = True
        return self._data

    def _read_and_compute(self, index: tuple) -> np.ndarray:
        """
        Helper to read minimal data and compute average.
        Result shape logic:
        If data is (x1, x2) and mft_axis=2 (x2), result is (x1, 1).
        User index maps to (t, x1_slice, dummy_slice).
        """
        # Normalize index to full tuple (t, s1, s2...)
        if not isinstance(index, tuple):
            index = (index,)

        # Separate time and spatial
        time_idx = index[0]
        user_spatial = index[1:]

        # 1. Determine Slices for Reading
        # We must read full MFT axis. Other axes follow user request.
        # Note: User request indices map to the NON-MFT axes in order.

        # Physical axes (1-based)
        all_axes = list(range(1, self._dim + 1))
        # Axes that are preserved in the result (before the final dummy dim)
        preserved_axes = [ax for ax in all_axes if ax != self._mft_axis]

        read_slices = [slice(None)] * self._dim  # For spatial dims only (0-based list)
        squeeze_map = []  # To track squeezes on the result

        # Helper to process slice/int
        def process_idx(idx, max_len):
            if isinstance(idx, int):
                if idx < 0:
                    idx += max_len
                return slice(idx, idx + 1), True
            return idx, False

        # Handle Time
        t_slice, t_sq = process_idx(time_idx, self._diag._maxiter)
        squeeze_map.append(t_sq)  # 0 for Time

        # Handle Spatial
        # We need to map user_spatial[i] to the correct physical read_slice

        # Also need to handle the dummy dimension at the end of user request
        # The MFT result has shape: [preserved_dims..., 1]

        nx_arr = self._diag._nx
        if self._dim == 1 and np.isscalar(nx_arr):
            nx_arr = [nx_arr]

        crop_final = []  # Slices to apply to the COMPUTED result

        # Iterate over preserved axes (these map to user's first N-1 spatial indices)
        for i, ax in enumerate(preserved_axes):
            # 0-based index in read_slices/nx_arr
            phys_idx = ax - 1
            max_len = nx_arr[phys_idx]

            if i < len(user_spatial):
                u_idx = user_spatial[i]
                sl, sq = process_idx(u_idx, max_len)
                read_slices[phys_idx] = sl
                # For the result, we don't crop again because we read exact region (except int->slice)
                # But we do need to squeeze if int was passed
                squeeze_map.append(sq)
            else:
                squeeze_map.append(False)  # No user constraint, so keep full

        # The MFT axis must be read fully
        # read_slices[self._mft_axis - 1] is already slice(None)

        # Handle the trailing dummy dimension in user request
        # Result has shape (..., 1). User might have indexed it.
        dummy_sq = False
        if len(user_spatial) > len(preserved_axes):
            # The user provided an index for the dummy dimension
            dummy_idx = user_spatial[len(preserved_axes)]
            # It acts on a dimension of size 1
            if isinstance(dummy_idx, int):
                # Typically index 0 or -1 is valid
                if dummy_idx not in [0, -1]:
                    # technically out of bounds if not 0, but let numpy handle or ignore
                    pass
                dummy_sq = True
            # slice is no-op
        squeeze_map.append(dummy_sq)

        # 2. Read Data
        # Diagnostic expects (time, spatial...)
        full_read_slices = tuple([t_slice] + read_slices)
        data = self._diag[full_read_slices]

        # 3. Compute Mean
        # data shape: (t_chunk, x1_chunk, x2_chunk...)
        # We forced mft_axis to be full. Others are sliced.
        # Which axis is mft_axis in 'data'?
        # Since we used slices (not ints) for reading, dimensions are preserved relative to Diag structure.
        # Diag (usually) returns (Time, X1, X2...).
        # So mft_axis corresponds to axis index 'mft_axis' (since Time is 0).

        # Compute mean
        # Result reduces dimension 'mft_axis'
        avg = data.mean(axis=self._mft_axis)

        # Expand dims at the end
        res = np.expand_dims(avg, axis=-1)

        # 4. Squeeze
        # Squeeze map corresponds to [Time, Preserved_1, Preserved_2, ..., Dummy]
        for i in range(len(squeeze_map) - 1, -1, -1):
            if squeeze_map[i]:
                res = res.squeeze(axis=i)

        return res

    def _data_generator(self, index, data_slice=None):
        """Generate average data for a specific index"""
        # Adapt to new signature standard or keep compatibility?
        # The base implementation used index (int) and yielded.
        # We relay to _read_and_compute
        if data_slice is None:
            # Full slice implies empty tuple of extra constraints?
            # Or full slice?
            # _read_and_compute expects (index, ...slices).
            yield self._read_and_compute((index,))
        else:
            yield self._read_and_compute((index, *data_slice))

    def __getitem__(self, index):
        """Get average at a specific index"""
        if self._all_loaded and self._data is not None:
            return self._data[index]

        return self._read_and_compute(index)


class MFT_Diagnostic_Fluctuations(Diagnostic):
    """
    Class to compute the fluctuation component of mean field theory.
    Inherits from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    mft_axis : int
        The axis to compute the mean field theory.

    """

    def __init__(self, diagnostic, mft_axis):
        # Initialize with the same species as the diagnostic
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=(diagnostic._simulation_folder if hasattr(diagnostic, "_simulation_folder") else None),
                species=diagnostic._species,
            )
        else:
            super().__init__(None)

        if mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")

        self.postprocess_name = "MFT_FLT"

        self._name = f"MFT_delta[{diagnostic._name}, {mft_axis}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False

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

    def load_all(self):
        """Load all data and compute the fluctuations"""
        if self._diag._all_loaded is True:
            print("Diagnostic data already loaded ... applyting MFT")
            self._data = self._diag._data
        if self._data is not None:
            print("Data already loaded")
            return self._data

        if not hasattr(self._diag, "_data") or self._diag._data is None:
            self._diag.load_all()

        if self._mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")
        else:
            # Compute the average
            avg = self._diag._data.mean(axis=self._mft_axis)
            # Reshape avg for broadcasting
            broadcast_shape = list(self._diag._data.shape)
            broadcast_shape[self._mft_axis] = 1
            avg_reshaped = avg.reshape(broadcast_shape)
            # Compute the fluctuations
            self._data = self._diag._data - avg_reshaped

        self._all_loaded = True
        return self._data

    def _read_and_compute(self, index: tuple) -> np.ndarray:
        """
        Helper to read minimal data (superset for MFT) and compute fluctuations.
        Result shape is same as original data.
        """
        if not isinstance(index, tuple):
            index = (index,)

        time_idx = index[0]
        user_spatial = index[1:]

        # 1. Determine Read Slices
        # We must read full MFT axis to compute average.
        # For other axes, we respect user slices.
        # If user sliced MFT axis, we read FULL, but must CROP result later.

        read_slices = [slice(None)] * self._dim
        crop_final_slices = [slice(None)] * (self._dim + 1)  # +1 for Time
        squeeze_map = []  # [Time, x1, x2...]

        nx_arr = self._diag._nx
        if self._dim == 1 and np.isscalar(nx_arr):
            nx_arr = [nx_arr]

        # Process Time
        def process_idx(idx, max_len):
            if isinstance(idx, int):
                if idx < 0:
                    idx += max_len
                # Read as slice to preserve dim
                return slice(idx, idx + 1), True, slice(None)  # No crop needed on result axis 0 as we read 1
            else:
                return idx, False, slice(None)

        t_sl, t_sq, t_crop = process_idx(time_idx, self._diag._maxiter)
        squeeze_map.append(t_sq)
        crop_final_slices[0] = t_crop

        # Process Spatial
        for i in range(self._dim):  # 0 to dim-1
            axis_1based = i + 1
            max_len = nx_arr[i]

            # Helper to calculate crop if we over-read
            def get_crop_for_overread(user_sl, max_l):
                # We read [0:max]. User wanted user_sl.
                # So on the result (which is size max), we apply user_sl.
                return user_sl

            if i < len(user_spatial):
                u_idx = user_spatial[i]

                if axis_1based == self._mft_axis:
                    # MUST READ FULL
                    read_slices[i] = slice(None)

                    if isinstance(u_idx, int):
                        if u_idx < 0:
                            u_idx += max_len
                        # We will read full, so result has size max_len at this axis.
                        # We want index u_idx.
                        crop_final_slices[axis_1based] = slice(u_idx, u_idx + 1)
                        squeeze_map.append(True)
                    else:
                        # User passed slice. Apply to final result.
                        crop_final_slices[axis_1based] = u_idx
                        squeeze_map.append(False)
                else:
                    # Can read partial
                    sl, sq, _ = process_idx(u_idx, max_len)
                    read_slices[i] = sl
                    squeeze_map.append(sq)
                    # No crop needed as we read exact
            else:
                # No user constraint
                squeeze_map.append(False)

        # 2. Read Data
        full_read_slices = tuple([t_sl] + read_slices)
        data = self._diag[full_read_slices]
        # data shape: (t_chunk, x1_chunk, x2_chunk...)
        # Note: MFT axis is FULL size. Others are sliced size.

        # 3. Compute Mean
        # MFT axis is self._mft_axis (0 is Time).
        avg = data.mean(axis=self._mft_axis)

        # Reshape avg for broadcasting
        # broadcast_shape = list(data.shape)
        # broadcast_shape[self._mft_axis] = 1
        # avg_reshaped = avg.reshape(broadcast_shape)
        # cleaner:
        avg_reshaped = np.expand_dims(avg, axis=self._mft_axis)

        # 4. Compute Fluctuations
        res = data - avg_reshaped

        # 5. Crop
        # If we over-read the MFT axis, we apply the crop now
        res = res[tuple(crop_final_slices)]

        # 6. Squeeze
        for i in range(len(squeeze_map) - 1, -1, -1):
            if squeeze_map[i]:
                res = res.squeeze(axis=i)

        return res

    def _data_generator(self, index, data_slice=None):
        """Generate fluctuation data for a specific index"""
        if data_slice is None:
            yield self._read_and_compute((index,))
        else:
            yield self._read_and_compute((index, *data_slice))

    def __getitem__(self, index):
        """Get fluctuations at a specific index"""
        if self._all_loaded and self._data is not None:
            return self._data[index]

        return self._read_and_compute(index)


class MFT_Species_Handler:
    """
    Class to handle mean field theory for a species.
    Acts as a wrapper for the MFT_Diagnostic class.

    Not intended to be used directly, but through the MFT_Simulation class.

    Parameters
    ----------
    species_handler : Species_Handler
        The species handler object.
    mft_axis : int
        The axis to compute the mean field theory.
    """

    def __init__(self, species_handler, mft_axis):
        self._species_handler = species_handler
        self._mft_axis = mft_axis
        self._mft_computed = {}

    def __getitem__(self, key):
        if key not in self._mft_computed:
            diag = self._species_handler[key]
            self._mft_computed[key] = MFT_Diagnostic(diag, self._mft_axis)
        return self._mft_computed[key]
