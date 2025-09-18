# Try to import "osiris_utils" package
import os

import numpy as np
import tqdm as tqdm

import osiris_utils as ou


class DatabaseCreator:
    """
    Class to create a database from a OSIRIS simulation
    """

    def __init__(self, simulation, species, save_folder):
        """
        Initialize the DatabaseCreator with a simulation, species, and save folder.

        Parameters
        ----------
        simulation : osiris_utils.Simulation
            The OSIRIS simulation object containing the data.
        species : str
            The species for which the database is being created (e.g., "electrons").
        save_folder : str
            The folder where the database will be saved.
        """

        self.simulation = simulation
        self.species = species
        self.save_folder = save_folder

        self.F_in = 22
        self.F_out = 1

    def set_limits(self, initial_iter=0, final_iter=None):
        self.initial_iter = initial_iter
        self.final_iter = final_iter
        self.T = final_iter - initial_iter
        self.X = self.simulation["e1"].nx[0]

    def _input_database(self, name):
        d_dx1 = ou.Derivative_Simulation(self.simulation, "x1")
        d_dt = ou.Derivative_Simulation(self.simulation, "t")

        self.simulation[self.species].add_diagnostic(self.simulation[self.species]["n"] * self.simulation[self.species]["T11"], "nT11")
        self.simulation[self.species].add_diagnostic(self.simulation[self.species]["n"] * self.simulation[self.species]["T12"], "nT12")

        # First derivatives
        self.simulation[self.species].add_diagnostic(ou.Derivative_Diagnostic(self.simulation[self.species]["nT11"], "x1"), "dnT11_dx1")
        self.simulation[self.species].add_diagnostic(ou.Derivative_Diagnostic(self.simulation[self.species]["nT12"], "x2"), "dnT12_dx2")
        self.simulation[self.species].add_diagnostic(d_dt[self.species]["vfl1"], "dvfl1_dt")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["vfl1"], "dvfl1_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["vfl2"], "dvfl2_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["vfl3"], "dvfl3_dx1")
        self.simulation.add_diagnostic(d_dx1["b2"], "db2_dx1")
        self.simulation.add_diagnostic(d_dx1["b3"], "db3_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["n"], "dn_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["T11"], "dT11_dx1")

        d_dx1 = ou.Derivative_Simulation(self.simulation, "x1")

        # Second derivatives
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["dnT11_dx1"], "d2_nT11_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["dvfl1_dx1"], "d2_vfl1_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["dvfl2_dx1"], "d2_vfl2_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["dvfl3_dx1"], "d2_vfl3_dx1")
        self.simulation.add_diagnostic(d_dx1["db2_dx1"], "d2_b2_dx1")
        self.simulation.add_diagnostic(d_dx1["db3_dx1"], "d2_b3_dx1")
        self.simulation[self.species].add_diagnostic(d_dx1[self.species]["dn_dx1"], "d2_n_dx1")

        sim_mft = ou.MFT_Simulation(self.simulation, mft_axis=2)

        dnT11_dx_avg = ou.Derivative_Diagnostic(sim_mft[self.species]["n"]["avg"] * sim_mft[self.species]["T11"]["avg"], "x1")

        data_array = np.empty((int(self.T), int(self.F_in), int(self.X)), dtype=np.float32)

        for i, t_idx in enumerate(tqdm.tqdm(range(self.initial_iter, self.final_iter))):
            feature_list = [
                # sim_mft["e_vlasov"]["avg"][t_idx].flatten(),
                sim_mft["b2"]["avg"][t_idx].flatten(),
                sim_mft["b3"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["vfl1"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["vfl2"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["vfl3"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["n"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["T11"]["avg"][t_idx].flatten(),
                sim_mft[self.species]["T12"]["avg"][t_idx].flatten(),
                # sim_mft[self.species]["dvfl1_dt"]["avg"][t_idx].flatten(),
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

            # Stack features: shape = (F, X)
            data_array[i] = np.stack(feature_list)

        np.save(os.path.join(self.save_folder, f"{name}.npy"), data_array)

        del data_array

    def _validate_and_clean_data(self, data):
        """
        Check for NaN and inf values and replace them with zeros.

        Parameters
        ----------
        data : np.ndarray
            The data array to validate and clean.

        Returns
        -------
        np.ndarray
            The cleaned data array with NaN/inf values replaced by zeros.
        """
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))

        if nan_count > 0 or inf_count > 0:
            print(f"Warning: Found {nan_count} NaN and {inf_count} inf values. Replacing with zeros.")

        return np.where(np.isfinite(data), data, 0.0)

    def _output_database(self, name):
        conf = ou.AnomalousResistivityConfig(
            species=self.species, include_time_derivative=False, include_convection=True, include_pressure=True, include_magnetic_force=True
        )
        self.ar = ou.AnomalousResistivity(self.simulation, self.species, config=conf)
        # Pre-allocate array: shape = [T, F, X]
        data_array_output = np.empty((int(self.T), int(self.F_out), int(self.X)), dtype=np.float32)

        for i, t_idx in enumerate(tqdm.tqdm(range(self.initial_iter, self.final_iter))):
            feature_list = [
                self.ar["eta"][t_idx].flatten(),
                # self.ar["e_vlasov_avg"][t_idx].flatten(),
            ]

            # Stack features: shape = (F, X)
            data_array_output[i] = np.stack(feature_list)

        # Validate and clean the entire array at once
        data_array_output = self._validate_and_clean_data(data_array_output)
        np.save(os.path.join(self.save_folder, f"{name}.npy"), data_array_output)

        del data_array_output

    def _E_vlasov_database(self, name):
        conf = ou.AnomalousResistivityConfig(
            species=self.species, include_time_derivative=False, include_convection=True, include_pressure=True, include_magnetic_force=True
        )
        self.ar = ou.AnomalousResistivity(self.simulation, self.species, config=conf)
        # Pre-allocate array: shape = [T, F, X]
        data_array_output = np.empty((int(self.T), 1, int(self.X)), dtype=np.float32)

        for i, t_idx in enumerate(tqdm.tqdm(range(self.initial_iter, self.final_iter))):
            feature_list = [self.ar["e_vlasov_avg"][t_idx].flatten()]

            # Stack features: shape = (F, X)
            data_array_output[i] = np.stack(feature_list)

        # Validate and clean the entire array at once
        data_array_output = self._validate_and_clean_data(data_array_output)
        np.save(os.path.join(self.save_folder, f"{name}.npy"), data_array_output)

        del data_array_output

    def create_database(self, database="both", name_input="input_tensor", name_output="eta_tensor", vlasov_name="e_vlasov_tensor"):
        """
        Create the input and output databases.

        Parameters
        ----------
        database : str
            The type of database to create ("input", "output", or "both").
        name_input : str
            The name of the input file (without extension). Default is "input_tensor".
        name_output : str
            The name of the output file (without extension). Default is "eta_tensor".

        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

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
        else:
            raise ValueError("Invalid database type. Choose 'input', 'output', or 'both'.")

        print(f"Databases created and saved in {self.save_folder}")
