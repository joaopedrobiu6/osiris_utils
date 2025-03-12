import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from .utils import *
from .data import *
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import tqdm as tqdm


class AnomalousResistivity:
    def __init__(self, quantity_folder, velocity_folder, quantity_iter, dump):
        
        
        # Load the quantities
        iter = quantity_iter
        vt_minus = dump*iter - 1
        vt_plus = dump*iter + 1
        
        self.E1 = OsirisGridFile(quantity_folder + f'FLD/e1/e1-{iter:06}.h5')
        self.B2 = OsirisGridFile(quantity_folder + f'FLD/b2/b2-{iter:06}.h5')
        self.B3 = OsirisGridFile(quantity_folder + f'FLD/b3/b3-{iter:06}.h5')
        
        self.V1 = OsirisGridFile(quantity_folder + f'UDIST/electrons/vfl1/vfl1-electrons-{iter:06}.h5')
        self.V2 = OsirisGridFile(quantity_folder + f'UDIST/electrons/vfl2/vfl2-electrons-{iter:06}.h5')
        self.V3 = OsirisGridFile(quantity_folder + f'UDIST/electrons/vfl3/vfl3-electrons-{iter:06}.h5')

        self.V1_b = OsirisGridFile(velocity_folder + f"vfl1-electrons-{vt_minus:06}.h5")
        self.V1_a = OsirisGridFile(velocity_folder + f"vfl1-electrons-{vt_plus:06}.h5")

        self.ne = OsirisGridFile(quantity_folder + f'DENSITY/electrons/charge/charge-electrons-{iter:06}.h5')
        self.ne.data = -self.ne.data
        
        self.T11 = OsirisGridFile(quantity_folder + f'UDIST/electrons/T11/T11-electrons-{iter:06}.h5')
        self.T12 = OsirisGridFile(quantity_folder + f'UDIST/electrons/T12/T12-electrons-{iter:06}.h5')
        self.P11 = OsirisGridFile(quantity_folder + f'UDIST/electrons/P11/P11-electrons-{iter:06}.h5')
        self.P11.data = self.P11.data - self.ne.data*self.V1.data*self.V1.data
        
        # Compute components of the mometum equation
        self.dV1dt = (self.V1_a.data - self.V1_b.data)/(2*self.V1_a.dt)
        self.V1_dV1dx = self.V1.data * np.gradient(self.V1.data, self.V1.dx[0], axis=0)
        
        self.dT11nedx = np.gradient(self.T11.data*self.ne.data, self.T11.dx[0], axis=0)
        self.dT12nedy = np.gradient(self.T12.data*self.ne.data, self.T12.dx[1], axis=1)
        
        self.V2B3 = self.V2.data * self.B3.data
        self.V3B2 = self.V3.data * self.B2.data  
        
        self.E_vlasov = - self.dV1dt - self.V1_dV1dx - (1/self.ne.data)*(self.dT11nedx + self.dT12nedy) - (self.V2B3 - self.V3B2)

        # Separate quantities in average and fluctuating parts - A = A_bar + A_delta
        self.V1_b_bar = np.expand_dims(transverse_average(self.V1_b.data), axis=1)
        self.V1_a_bar = np.expand_dims(transverse_average(self.V1_a.data), axis=1)

        self.T11_bar = np.expand_dims(transverse_average(self.T11.data), axis=1)
        self.T11_delta = (self.T11.data - self.T11_bar)
        
        self.T12_bar = np.expand_dims(transverse_average(self.T12.data), axis=1)
        self.T12_delta = (self.T12.data - self.T12_bar)

        self.ne_bar = np.expand_dims(transverse_average(self.ne.data), axis=1)
        self.ne_delta = (self.ne.data - self.ne_bar)
        
        self.B2_bar = np.expand_dims(transverse_average(self.B2.data), axis=1)
        self.B2_delta = (self.B2.data - self.B2_bar)
        self.B3_bar = np.expand_dims(transverse_average(self.B3.data), axis=1)
        self.B3_delta = (self.B3.data - self.B3_bar)
        
        self.V1_bar = np.expand_dims(transverse_average(self.V1.data), axis=1)
        self.V1_delta = (self.V1.data - self.V1_bar)
        self.V2_bar = np.expand_dims(transverse_average(self.V2.data), axis=1)
        self.V2_delta = (self.V2.data - self.V2_bar)
        self.V3_bar = np.expand_dims(transverse_average(self.V3.data), axis=1)
        self.V3_delta = (self.V3.data - self.V3_bar)
        
        self.E_vlasov_bar = np.expand_dims(transverse_average(self.E_vlasov), axis=1)

        # Compute components of the mometum equation for average quantities
        self.dV1dt_bar = (self.V1_a_bar - self.V1_b_bar)/(2*self.V1_a.dt)
        self.V1_dV1dx_bar = self.V1_bar*np.gradient(self.V1_bar, self.V1.dx[0], axis=0)
        
        self.dT11nedx_bar = np.gradient(self.T11_bar * self.ne_bar, self.T11.dx[0], axis=0)
        self.dT11nedx_delta = np.gradient(self.T11_delta * self.ne_delta, self.T11.dx[0], axis=0)
            
        self.V2V3_bar = self.V2_bar*self.B3_bar
        self.V3V2_bar = self.V3_bar*self.B2_bar

        # Momentum equation for average quantities
        self.momentum_bar = self.E_vlasov_bar + self.dV1dt_bar + self.V1_dV1dx_bar + self.dT11nedx_bar/self.ne_bar + self.V2V3_bar - self.V3V2_bar         
        self.momentum_bar = np.squeeze(self.momentum_bar)
        
        # Compute the anomalous resistivity - averages over second-order non vanishing transverse fluctuations
        
        # convective term, u x B term - common terms
        self.term1 = transverse_average(self.V1_delta*np.gradient(self.V1_delta, self.V1.dx[0], axis=0))                                                            # -
        self.term2 = transverse_average(self.V2_delta*self.B3_delta)                                                                                                # -
        self.term3 = transverse_average(self.V3_delta*self.B2_delta)                                                                                                # + 
        
        self.commom_terms = - self.term1 - self.term2 + self.term3

        # Thermal pressure gradient term xx 
        # Model XX1
        self.xx1_term1 = transverse_average((np.gradient(self.ne_bar*self.T11_bar, self.T11.dx[0], axis=0)/self.ne.data)*(self.ne_delta/self.ne_bar))               # +
        self.xx1_term2 = transverse_average(np.gradient(self.ne_bar*self.T11_delta, self.T11.dx[0], axis=0)/self.ne.data)                                           # -
        self.xx1_term3 = transverse_average(np.gradient(self.ne_delta*self.T11_bar, self.T11.dx[0], axis=0)/self.ne.data)                                           # -
        self.xx1_term4 = transverse_average(np.gradient(self.ne_delta*self.T11_delta, self.T11.dx[0], axis=0)/self.ne.data)                                         # -

        self.xx1_full = self.xx1_term1 - self.xx1_term2 - self.xx1_term3 - self.xx1_term4

        # Model XX2
        self.xx2_term1 = transverse_average(np.gradient(self.ne_delta*self.T11_delta, self.T11.dx[0], axis=0)/self.ne_bar)                                          # -
        self.xx2_term2 = transverse_average((np.gradient(self.ne.data*self.T11.data, self.T11.dx[0], axis=0)/self.ne.data)*(self.ne_delta/self.ne_bar))             # +

        self.xx2_full = - self.xx2_term1 + self.xx2_term2

        # Thermal pressure gradient term xy
        # Model XY1
        self.xy1_term1 = transverse_average(np.gradient(self.ne_bar*self.T12_delta, self.T12.dx[1], axis=1)/self.ne.data)                                           # -
        self.xy1_term2 = transverse_average(np.gradient(self.ne_delta*self.T12_bar, self.T12.dx[1], axis=1)/self.ne.data)                                           # -
        self.xy1_term3 = transverse_average(np.gradient(self.ne_delta*self.T12_delta, self.T12.dx[1], axis=1)/self.ne.data)                                         # -
        
        self.xy1_full = - self.xy1_term1 - self.xy1_term2 - self.xy1_term3
        
        # Model XY2
        self.xy2_term1 = transverse_average(np.gradient(self.ne_delta*self.T12_delta, self.T12.dx[1], axis=1)/self.ne_bar)                                          # -
        self.xy2_term2 = transverse_average((np.gradient(self.ne.data*self.T12.data, self.T12.dx[1], axis=1)/self.ne.data)*(self.ne_delta/self.ne_bar))             # +

        self.xy2_full = - self.xy2_term1 + self.xy2_term2

        # self.eta =  - self.term1 - self.term2 + self.term3 + self.term4 - self.term5 - self.term6 - self.term7 - self.term8 - self.term9 - self.term10        
        # self.eta_dominant = - self.term2 + self.term4 - self.term5 - self.term6 - self.term7 - self.term8 - self.term10
        # self.eta_model1 = - self.term1 - self.term2 + self.term3 - self.term4_model1 + self.term5_model1 - self.term8 - self.term9 - self.term10 
        # self.eta_model1_dom = - self.term2 - self.term4_model1 + self.term5_model1 - self.term8 - self.term10 
        
        self.eta_xx1_xy1 = self.commom_terms + self.xx1_full + self.xy1_full
        self.eta_xx2_xy2 = self.commom_terms + self.xx2_full + self.xy2_full
        self.eta_xx1_xy2 = self.commom_terms + self.xx1_full + self.xy2_full
        self.eta_xx2_xy1 = self.commom_terms + self.xx2_full + self.xy1_full

        self.eta = self.eta_xx1_xy1
        self.eta_dominant = - self.term2 + self.xx1_term1 - self.xx1_term2 - self.xx1_term3 - self.xx1_term4 - self.xy1_term1 - self.xy1_term3



    def Momentum(self) -> tuple:
        """
        Returns the momentum equation for average quantities, eta, and the eta with dominant terms

        Returns:
        --------
        momentum_bar: numpy array
            momentum equation for average quantities
        eta: numpy array
            eta term in the momentum equation
        eta_dominant: numpy array
            eta term with dominant terms
        """
        return self.momentum_bar, self.eta, self.eta_dominant
    
    def MomentumTerms(self) -> tuple:
        """ 
        Returns the terms of the momentum equation

        Returns:
        --------
        - E_vlasov_bar: numpy array
            sign +
        - dV1dt_bar: numpy array
            sign +
        - V1_dV1dx_bar: numpy array
            sign +
        - dT11nedx_bar/ne_bar: numpy array
            sign +
        - V2V3_bar: numpy array
            sign +
        - V3V2_bar: numpy array
            sign -
        """
        return np.squeeze(self.E_vlasov_bar), np.squeeze(self.dV1dt_bar), np.squeeze(self.V1_dV1dx_bar), np.squeeze(self.dT11nedx_bar/self.ne_bar), np.squeeze(self.V2V3_bar), np.squeeze(self.V3V2_bar) 
    
    def model_xx1_xy1(self):
        return self.momentum_bar, self.eta_xx1_xy1
    
    def model_xx2_xy2(self):
        return self.momentum_bar, self.eta_xx2_xy2
    
    def model_xx1_xy2(self):
        return self.momentum_bar, self.eta_xx1_xy2
    
    def model_xx2_xy1(self):
        return self.momentum_bar, self.eta_xx2_xy1
    
    def ElectricFields(self):
        return self.E1, self.E_vlasov
    
    def SaveDatabase(self, filename):
        dataframe = pd.DataFrame({'T11_bar': np.squeeze(self.T11_bar), 'T12_bar': np.squeeze(self.T12_bar), 'ne_bar': np.squeeze(self.ne_bar),
                          'V1_bar': np.squeeze(self.V1_bar), 'V2_bar': np.squeeze(self.V2_bar), 'V3_bar': np.squeeze(self.V3_bar),
                          'B2_bar': np.squeeze(self.B2_bar), 'B3_bar': np.squeeze(self.B3_bar), 'E_vlasov': np.squeeze(self.E_vlasov_bar),
                          'dT11nedx_bar': np.squeeze(self.dT11nedx_bar), 'dT12nedy_bar': np.squeeze(self.dT11nedx_bar),
                          'eta': np.squeeze(self.eta)})
        dataframe_table = pa.Table.from_pandas(dataframe)
        pq.write_table(dataframe_table, filename)
    
    def Axis_to_Plot(self):
        x = np.arange(self.ne.grid[0][0], self.ne.grid[0][1], self.ne.dx[0])
        dx = self.ne.dx[0]
        return x, dx

def Omega_K(E_filename, eta_filename, quantities_folder, velocity_folder, range_iter, dump):
    
    def load_and_create_file(filename, func="ElectricFields"):
        # print loading data
        print(f"Loading data for {filename}...")
        try:
            data = np.loadtxt(filename + ".txt")
        except:
            data_list = []
            for i in tqdm.trange(range_iter[0], range_iter[1], desc="Computing " + filename):
                if func == "ElectricFields":
                    data_list.append(transverse_average(AnomalousResistivity(quantities_folder, velocity_folder, i, dump).ElectricFields()[1]))
                elif func == "Momentum":
                    data_list.append(AnomalousResistivity(quantities_folder, velocity_folder, i, dump).Momentum()[1])
            data = np.nan_to_num(data_list)
        data = np.array(data)

        if np.isnan(data).any():
            data = np.nan_to_num(data)
        if np.isinf(data).any():
            data = np.nan_to_num(data)
        if not os.path.isfile(filename + ".txt"):
            np.savetxt(filename + ".txt", data)
            print(f"Data saved as {filename}.txt...")
        return data
        

    def omegak(data):
        print("Computing the FFT of the data...")
        E1 = OsirisGridFile(quantities_folder + f'FLD/e1/e1-000001.h5')
        data_han = data * np.hanning(data.shape[0])[:, np.newaxis]
        # data_fft = np.abs(np.fft.fft2(data_han))**2
        # data_fft = np.fft.fftshift(data_fft)

        # # Compute the magnitude of the FFT
        # magnitude = data_fft

        # # Compute the corresponding k and omega values
        # kx = np.fft.fftfreq(data.shape[1], d=E1.dx[0])  # Wavenumbers (k)
        # omega = np.fft.fftfreq(data.shape[0], d=E1.dt)  # Frequencies (ω)

        # # Shift the frequencies to match the shifted FFT
        # kx_shifted = np.fft.fftshift(kx)
        # omega_shifted = np.fft.fftshift(omega)

        # # Create a meshgrid for k and omega
        # K, W = np.meshgrid(kx_shifted, omega_shifted)
        # print("FFT computed...")
        # return K, W, magnitude, kx_shifted

        sp = np.abs(np.fft.fft2(data_han))**2
        sp = np.fft.fftshift( sp )

        k_max = np.pi / E1.dx[0]
        omega_max = np.pi / E1.dt

        return k_max, omega_max, sp
    
    def plot_dispersion_relation(k_max, omega_max, magnitude, normalized=True, filename="DispersionRelation.png"):
        print("Plotting the dispersion relation...")
        print("Shape of the magnitude: ", magnitude.shape)
        if normalized:
            magnitude = magnitude/magnitude.max()

        v_the = 0.001
        k = np.linspace(-k_max, k_max, num = magnitude.shape[1])
        w=np.sqrt(1 + 3 * v_the**2 * k**2)

        fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
        im = ax.imshow( magnitude, origin = 'lower', extent = ( -k_max, k_max, -omega_max, omega_max ),
           aspect = 'auto', cmap = 'gray')#, norm = LogNorm())
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$|A(k, \omega)|^2$')
        # ax.plot(k, w, label=r'$\omega(k) = \sqrt{\omega_p^2 + 3v_{th}^2 k^2}$')
        ax.set_title(r'$|A(k, \omega)|^2$')
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$\omega$')
        ax.set_xlim(0, k_max)
        ax.set_ylim(0, omega_max)
        # ax.legend()

        return fig, ax

    E = load_and_create_file(E_filename, "ElectricFields")
    AR = load_and_create_file(eta_filename, "Momentum")

    E_omegak = omegak(E)
    AR_omegak = omegak(AR)

    fig_e, ax_e = plot_dispersion_relation(*E_omegak, normalized=False)
    fig_ar, ax_ar = plot_dispersion_relation(*AR_omegak, normalized=False)

    fig_e.savefig(E_filename + ".png")
    fig_ar.savefig(eta_filename + ".png")

    print("Done!")
    return E_omegak, AR_omegak

    # For labels and infos:

