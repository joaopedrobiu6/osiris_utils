from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import tqdm as tqdm

from ..ar import AnomalousResistivity, AnomalousResistivityConfig
from ..postprocessing import Derivative_Diagnostic, Derivative_Simulation, MFT_Simulation

__all__ = ["DatabaseBuildConfig", "DatabaseCreator"]


@dataclass(frozen=True)
class DatabaseBuildConfig:
    """Configuration for database tensor creation."""

    dtype: type = np.float32
    max_workers: int | None = None  # None -> ThreadPoolExecutor default


class DatabaseCreator:
    """
    Class to create a database from a OSIRIS simulation
    """

    def __init__(self, simulation, species: str, save_folder: str, build_config: DatabaseBuildConfig | None = None):
        self.simulation = simulation
        self.species = species
        self.save_folder = save_folder
        self.build_config = build_config or DatabaseBuildConfig()

        # Default feature counts (kept for compatibility)
        self.F_in = 22
        self.F_out = 1

        # Limits (set by set_limits)
        self.initial_iter = 0
        self.final_iter = None
        self.T = 0
        self.X = 0

    def set_limits(self, initial_iter: int = 0, final_iter: int | None = None):
        """
        Set iteration range [initial_iter, final_iter) used for tensors.

        If final_iter is None, it will be inferred from the length of a diagnostic
        that should exist (e1 field), which matches prior behavior but safely.
        """
        self.initial_iter = int(initial_iter)

        if final_iter is None:
            # Infer time-length from a common field diagnostic (e1).
            # If your sim uses another canonical time diagnostic, swap it here.
            try:
                inferred = len(self.simulation["e1"])
            except Exception as e:
                raise ValueError("final_iter=None but could not infer simulation time length from simulation['e1'].") from e
            final_iter = inferred

        self.final_iter = int(final_iter)

        if self.final_iter <= self.initial_iter:
            raise ValueError(f"final_iter must be > initial_iter (got {self.final_iter} <= {self.initial_iter}).")

        self.T = self.final_iter - self.initial_iter

        try:
            self.X = int(self.simulation["e1"].nx[0])
        except Exception as e:
            raise ValueError("Could not determine spatial size X from simulation['e1'].nx[0].") from e

    def create_database(
        self,
        database: str = "both",
        name_input: str = "input_tensor",
        name_output: str = "eta_tensor",
        vlasov_name: str = "e_vlasov_tensor",
    ) -> None:
        """
        Create database tensors and save .npy files into save_folder.

        Parameters
        ----------
        database : str
            Database type. Can be 'both', 'input', 'output' and 'e_vlsov'.
        name_input : str
            Name for input database.
        name_output : str
            Name for output database.
        vlasov_name : str
            Name for database with "vlasov" electric field.
        """
        os.makedirs(self.save_folder, exist_ok=True)

        if self.final_iter is None:
            # Keep old behavior (but safer): infer if user forgot to call set_limits.
            self.set_limits(0, None)

        if database == "both":
            print("Creating input and output databases...")
            self._input_database(name=name_input)
            print("Input database created.")
            self._output_database(name=name_output)
            print("Output database created.")
        elif database == "input":
            print("Creating input database...")
            self._input_database(name=name_input)
            print("Input database created.")
        elif database == "output":
            print("Creating output database...")
            self._output_database(name=name_output)
            print("Output database created.")
        elif database == "e_vlasov":
            print("Creating E_vlasov database...")
            self._E_vlasov_database(name=vlasov_name)
            print("E_vlasov database created.")
        else:
            raise ValueError("Invalid database type. Choose 'input', 'output', 'both', or 'e_vlasov'.")

        print(f"Databases created and saved in {self.save_folder}")

    @staticmethod
    def _ensure_diagnostic(container, diagnostic, name: str) -> None:
        """
        Add a diagnostic only if it doesn't already exist.
        Works with both simulation-level and species-level containers.

        Parameters
        ----------
        container : dict-like
            The container to check/add the diagnostic to (e.g. simulation or species dict).
        diagnostic : Diagnostic object
            The diagnostic to add if not already present.
        """
        try:
            _ = container[name]
            return
        except Exception:
            container.add_diagnostic(diagnostic, name)

    @staticmethod
    def _validate_and_clean_data(data: np.ndarray) -> np.ndarray:
        """Replace NaN/inf with 0 (in-place) and return the same array.

        Parameters
        ----------
        data : np.ndarray
            The data array to validate and clean.

        Returns
        -------
        np.ndarray
            The cleaned data array with NaN/inf replaced by 0.
        """
        mask = ~np.isfinite(data)
        if np.any(mask):
            nan_count = int(np.count_nonzero(np.isnan(data)))
            inf_count = int(np.count_nonzero(np.isinf(data)))
            print(f"Warning: Found {nan_count} NaN and {inf_count} inf values. Replacing with zeros.")
            data[mask] = 0.0
        return data

    def _iter_range(self) -> Iterable[int]:
        return range(self.initial_iter, self.final_iter)

    def _build_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, int, int],
        frame_fn: Callable[[int], np.ndarray],
        desc: str,
        validate: bool = False,
    ):
        """
        Build a tensor by evaluating frame_fn(t_idx) for t_idx in [initial_iter, final_iter).
        Saves to {save_folder}/{name}.npy

        Parameters
        ----------
        name : str
            Name for the saved .npy file (without extension).
        shape : tuple[int, int, int]
            Shape of the tensor to build (T, F, X).
        frame_fn : Callable[[int], np.ndarray]
            Function that takes a time index and returns the corresponding frame as a numpy array of shape (F, X).
        desc : str
            
        """
        arr = np.empty(shape, dtype=self.build_config.dtype)

        if self.T <= 0:
            raise ValueError("Nothing to build: T <= 0. Did you call set_limits()?")

        # Warm-up: force initialization of diagnostics/etc on the main thread
        arr[0] = frame_fn(self.initial_iter)

        if self.T > 1:
            with ThreadPoolExecutor(max_workers=self.build_config.max_workers) as ex:
                futures = {
                    ex.submit(frame_fn, t_idx): out_i for out_i, t_idx in enumerate(range(self.initial_iter + 1, self.final_iter), start=1)
                }

                for fut in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    initial=1,
                    desc=desc,
                ):
                    out_i = futures[fut]
                    try:
                        arr[out_i] = fut.result()
                    except Exception as e:
                        raise RuntimeError(f"Failed while building '{name}' at output index {out_i}.") from e

        if validate:
            arr = self._validate_and_clean_data(arr)

        np.save(os.path.join(self.save_folder, f"{name}.npy"), arr)
        del arr

    def _input_database(self, name: str):
        # Derivative operators (create once)
        d_dx1 = Derivative_Simulation(self.simulation, "x1")
        d_dt = Derivative_Simulation(self.simulation, "t")

        sp = self.simulation[self.species]

        # Composite diagnostics
        self._ensure_diagnostic(sp, sp["n"] * sp["T11"], "nT11")
        self._ensure_diagnostic(sp, sp["n"] * sp["T12"], "nT12")

        # First derivatives
        self._ensure_diagnostic(sp, Derivative_Diagnostic(sp["nT11"], "x1"), "dnT11_dx1")
        self._ensure_diagnostic(sp, Derivative_Diagnostic(sp["nT12"], "x2"), "dnT12_dx2")
        self._ensure_diagnostic(sp, d_dt[self.species]["vfl1"], "dvfl1_dt")
        self._ensure_diagnostic(sp, d_dx1[self.species]["vfl1"], "dvfl1_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["vfl2"], "dvfl2_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["vfl3"], "dvfl3_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["b2"], "db2_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["b3"], "db3_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["n"], "dn_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["T11"], "dT11_dx1")

        # Second derivatives (reuse the same d_dx1 operator)
        self._ensure_diagnostic(sp, d_dx1[self.species]["dnT11_dx1"], "d2_nT11_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dvfl1_dx1"], "d2_vfl1_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dvfl2_dx1"], "d2_vfl2_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dvfl3_dx1"], "d2_vfl3_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["db2_dx1"], "d2_b2_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["db3_dx1"], "d2_b3_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dn_dx1"], "d2_n_dx1")

        # MFT wrapper
        sim_mft = MFT_Simulation(self.simulation, mft_axis=2)

        # This one is computed directly from averages (not added into sim dict)
        dnT11_dx_avg = Derivative_Diagnostic(sim_mft[self.species]["n"]["avg"] * sim_mft[self.species]["T11"]["avg"], "x1")

        def get_frame(t_idx: int) -> np.ndarray:
            # Keep the exact same feature ordering as your original code.
            feature_list = [
                sim_mft["b2"]["avg"][t_idx].flatten(),
                sim_mft["b3"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["vfl1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["vfl2"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["vfl3"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["n"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["T11"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["T12"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["dvfl1_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["dvfl2_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["dvfl3_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["dn_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["dT11_dx1"]["avg"][t_idx].flatten(),
                sim_mft["db2_dx1"]["avg"][t_idx].flatten(),
                sim_mft["db3_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["d2_vfl1_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["d2_vfl2_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["d2_vfl3_dx1"]["avg"][t_idx].flatten(),
                sim_mft["d2_b2_dx1"]["avg"][t_idx].flatten(),
                sim_mft["d2_b3_dx1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["d2_n_dx1"]["avg"][t_idx].flatten(),
                dnT11_dx_avg[t_idx].flatten(),
            ]
            stacked = np.stack(feature_list)
            if stacked.shape[0] != self.F_in:
                raise ValueError(f"Input feature count mismatch: expected {self.F_in}, got {stacked.shape[0]}.")
            return stacked

        self._build_tensor(
            name=name,
            shape=(int(self.T), int(self.F_in), int(self.X)),
            frame_fn=get_frame,
            desc="Creating Input Database",
            validate=False,
        )

    # ----------------------------
    # Output database (eta)
    # ----------------------------

    def _output_database(self, name: str):
        conf = AnomalousResistivityConfig(
            species=self.species,
            include_time_derivative=False,
            include_convection=True,
            include_pressure=True,
            include_magnetic_force=True,
        )
        ar = AnomalousResistivity(self.simulation, self.species, config=conf)

        def get_frame(t_idx: int) -> np.ndarray:
            feature_list = [
                ar["eta"][t_idx].flatten(),
            ]
            stacked = np.stack(feature_list)
            if stacked.shape[0] != self.F_out:
                raise ValueError(f"Output feature count mismatch: expected {self.F_out}, got {stacked.shape[0]}.")
            return stacked

        self._build_tensor(
            name=name,
            shape=(int(self.T), int(self.F_out), int(self.X)),
            frame_fn=get_frame,
            desc="Creating Output Database",
            validate=True,
        )

    def _E_vlasov_database(self, name: str):
        conf = AnomalousResistivityConfig(
            species=self.species,
            include_time_derivative=False,
            include_convection=True,
            include_pressure=True,
            include_magnetic_force=True,
        )
        ar = AnomalousResistivity(self.simulation, self.species, config=conf)

        def get_frame(t_idx: int) -> np.ndarray:
            feature_list = [ar["e_vlasov_avg"][t_idx].flatten()]
            return np.stack(feature_list)

        self._build_tensor(
            name=name,
            shape=(int(self.T), 1, int(self.X)),
            frame_fn=get_frame,
            desc="Creating E_vlasov Database",
            validate=True,
        )
