from .utils import transverse_average, integrate, open2D
import numpy as np
import pandas as pd
import scipy

class TwoFluid2D:
    def __init__(self, nx, ny, xmax, ymax, ppc, ndump, dt, time, root_folder, quantities_folder, velocity_folder):
        self.nx = nx
        self.ny = ny
        self.xmax = xmax
        self.ymax = ymax
        self.ppc = ppc
        self.ndump = ndump
        self.dt = dt
        self.time = time
        self.root_folder = root_folder
        self.quantities_folder = f"{self.root_folder}/{quantities_folder}"
        self.velocity_folder = f"{self.root_folder}/{velocity_folder}"
        
        self.vt = int(self.time/self.dt)
        self.qt = int(self.vt/self.ndump)
        self.dx = self.xmax/self.nx
        self.dy = self.ymax/self.ny
        
        self.qtime = f"{self.qt:06}"
        self.vtime_before1 = f"{(self.vt-1):06}"
        self.vtime_after1 = f"{(self.vt+1):06}"
        
        self.load_quantities()
        self.MomentumEquationQuantities_load()
        self.MeanFieldTheory_load()

    def load_quantities(self):
        # -------------------------------------------------------------------- FIELDS --------------------------------------------------------------------
        self.x, self.y, self.Ex, _ = open2D(f"{self.quantities_folder}/MS/FLD/e1/e1-{self.qtime}.h5")

        _, _, self.By, _ = open2D(f"{self.quantities_folder}/MS/FLD/b2/b2-{self.qtime}.h5")
        _, _, self.Bz, _ = open2D(f"{self.quantities_folder}/MS/FLD/b3/b3-{self.qtime}.h5")

        # ------------------------------------------------------------------- Electrons ------------------------------------------------------------------

        _, _, self.ve_x, _ = open2D(f"{self.quantities_folder}/MS/UDIST/electrons/vfl1/vfl1-electrons-{self.qtime}.h5")
        _, _, self.ve_y, _ = open2D(f"{self.quantities_folder}/MS/UDIST/electrons/vfl2/vfl2-electrons-{self.qtime}.h5")
        _, _, self.ve_z, _ = open2D(f"{self.quantities_folder}/MS/UDIST/electrons/vfl3/vfl3-electrons-{self.qtime}.h5")
        _, _, self.Pe_xx, _ = open2D(f"{self.quantities_folder}/MS/UDIST/electrons/P11/P11-electrons-{self.qtime}.h5", pressure=True)
        _, _, self.Pe_xy, _ = open2D(f"{self.quantities_folder}/MS/UDIST/electrons/P12/P12-electrons-{self.qtime}.h5", pressure=True)

        if self.ppc == 400:
            _, _, self.ve_x_before1, _ = open2D(f"{self.velocity_folder}/vfl1-electrons-{self.vtime_before1}.h5")
            _, _, self.ve_x_after1, _ = open2D(f"{self.velocity_folder}/vfl1-electrons-{self.vtime_after1}.h5")
        else:
            _, _, self.ve_x_before1, _ = open2D(f"{self.velocity_folder}/MS/UDIST/electrons/vfl1/vfl1-electrons-{self.vtime_before1}.h5")
            _, _, self.ve_x_after1, _ = open2D(f"{self.velocity_folder}/MS/UDIST/electrons/vfl1/vfl1-electrons-{self.vtime_after1}.h5")

        _, _, e_charge, _ = open2D(f"{self.quantities_folder}/MS/DENSITY/electrons/charge/charge-electrons-{self.qtime}.h5")
        self.ne = -e_charge

        # ------------------------------------------------------------------- Positrons ------------------------------------------------------------------

        _, _, self.vp_x, _ = open2D(f"{self.quantities_folder}/MS/UDIST/positrons/vfl1/vfl1-positrons-{self.qtime}.h5")
        _, _, self.vp_y, _ = open2D(f"{self.quantities_folder}/MS/UDIST/positrons/vfl2/vfl2-positrons-{self.qtime}.h5")
        _, _, self.vp_z, _ = open2D(f"{self.quantities_folder}/MS/UDIST/positrons/vfl3/vfl3-positrons-{self.qtime}.h5")
        _, _, self.Pp_xx, _ = open2D(f"{self.quantities_folder}/MS/UDIST/positrons/P11/P11-positrons-{self.qtime}.h5", pressure=True)
        _, _, self.Pp_xy, _ = open2D(f"{self.quantities_folder}/MS/UDIST/positrons/P12/P12-positrons-{self.qtime}.h5", pressure=True)

        if self.ppc == 400:
            _, _, self.vp_x_before1, _ = open2D(f"{self.velocity_folder}/vfl1-positrons-{self.vtime_before1}.h5")
            _, _, self.vp_x_after1, _ = open2D(f"{self.velocity_folder}/vfl1-positrons-{self.vtime_after1}.h5")
        else:
            _, _, self.vp_x_before1, _ = open2D(f"{self.velocity_folder}/MS/UDIST/positrons/vfl1/vfl1-positrons-{self.vtime_before1}.h5")
            _, _, self.vp_x_after1, _ = open2D(f"{self.velocity_folder}/MS/UDIST/positrons/vfl1/vfl1-positrons-{self.vtime_after1}.h5")

        _, _, charge_p, _ = open2D(f"{self.quantities_folder}/MS/DENSITY/positrons/charge/charge-positrons-{self.qtime}.h5")
        self.n_p = charge_p
        
    def MomentumEquationQuantities_load(self):
        self.dvdt_e = (self.ve_x_after1 - self.ve_x_before1) / (2*self.dt)
        # v.(dv/dx) for electrons
        self.vdvdx_e = self.ve_x * np.gradient(self.ve_x, self.dx, axis=1)
        # dP/dx and dP/dy for electrons
        self.dPdx_e = np.gradient(self.Pe_xx, self.dx, axis=1)
        self.dPdy_e = np.gradient(self.Pe_xy, self.dy, axis=0)
        # v x B for electrons
        self.vyBz_e = self.ve_y*self.Bz 
        self.vzBy_e = self.ve_z*self.By
        self.dvdt_p = (self.vp_x_after1 - self.vp_x_before1) / (2*self.dt)
        # v.(dv/dx) for positrons
        self.vdvdx_p = self.vp_x * np.gradient(self.vp_x, self.dx, axis=1)
        # dP/dx and dP/dy for positrons
        self.dPdx_p = np.gradient(self.Pp_xx, self.dx, axis=1)
        self.dPdy_p = np.gradient(self.Pp_xy, self.dy, axis=0)
        # v x B for positrons
        self.vyBz_p = self.vp_y*self.Bz 
        self.vzBy_p = self.vp_z*self.By
        
        
    def MomentumEquationQuantities(self, species):
        if species == 'electrons':
            dataframe_e = pd.DataFrame({'dvdt_e': self.dvdt_e, 'vdvdx_e': self.vdvdx_e, 'dPdx_e': self.dPdx_e, 'dPdy_e': self.dPdy_e, 'vyBz_e': self.vyBz_e, 
                                    'vzBy_e': self.vzBy_e, 'ne': self.ne, 'Ex': self.Ex, 'By': self.By, 'Bz': self.Bz})
            return dataframe_e
        elif species == 'positrons':
            dataframe_p = pd.DataFrame({'dvdt_p': self.dvdt_p, 'vdvdx_p': self.vdvdx_p, 'dPdx_p': self.dPdx_p, 'dPdy_p': self.dPdy_p, 'vyBz_p': self.vyBz_p, 
                                    'vzBy_p': self.vzBy_p, 'n_p': self.n_p, 'Ex': self.Ex, 'By': self.By, 'Bz': self.Bz})
            return dataframe_p
        
    def MomentumEquation(self, species):
        if species == 'electrons':
            SingleFluid_LHS_e_ = self.dvdt_e + self.vdvdx_e
            SingleFluid_RHS_e_ = -(self.dPdx_e +self. dPdy_e)/self.ne - (self.Ex + self.vyBz_e - self.vzBy_e)

            return transverse_average(SingleFluid_LHS_e_), transverse_average(SingleFluid_RHS_e_)
        
        elif species == 'positrons':
            SingleFluid_LHS_p_ = self.dvdt_p + self.vdvdx_p
            SingleFluid_RHS_p_ = -(self.dPdx_p + self.dPdy_p)/self.n_p + (self.Ex + self.vyBz_p - self.vzBy_p)
            
            return transverse_average(SingleFluid_LHS_p_), transverse_average(SingleFluid_RHS_p_)
    
    def AnomalousResistivity(self, species, n=False):
        if species == 'electrons':
            SingleFluid_RHS_e_ = - self.dvdt_e - self.vdvdx_e -(self.dPdx_e + self.dPdy_e)/self.ne - (self.vyBz_e - self.vzBy_e)
            if n:
                return transverse_average(self.Ex*self.ne), transverse_average(SingleFluid_RHS_e_ *self.ne)
            return transverse_average(self.Ex) ,transverse_average(SingleFluid_RHS_e_)
        
        elif species == 'positrons':
            SingleFluid_RHS_p_ = self.dvdt_p + self.vdvdx_p  + (self.dPdx_p + self.dPdy_p)/self.n_p - (self.vyBz_p - self.vzBy_p)
            if n:
                return transverse_average(self.Ex*self.n_p), transverse_average(SingleFluid_RHS_p_ *self.n_p)
            return transverse_average(self.Ex), transverse_average(SingleFluid_RHS_p_)
        
    def MeanFieldTheory_load(self):
        self.Ex_bar = transverse_average(self.Ex)
        self.Ex_delta = self.x - self.Ex_bar
        self.By_bar = transverse_average(self.By)
        self.By_delta = self.By - self.By_bar
        self.Bz_bar = transverse_average(self.Bz)
        self.Bz_delta = self.Bz - self.Bz_bar

        # ----------------------------------------------- ELECTRONS -----------------------------------------------
        self.ve_x_bar = transverse_average(self.ve_x)
        self.ve_x_delta = self.ve_x - self.ve_x_bar
        self.ve_y_bar = transverse_average(self.ve_y)
        self.ve_y_delta = self.ve_y - self.ve_y_bar
        self.ve_z_bar = transverse_average(self.ve_z)
        self.ve_z_delta = self.ve_z - self.ve_z_bar

        self.ve_x_before1_bar = transverse_average(self.ve_x_before1)
        self.ve_x_before1_delta = self.ve_x_before1 - self.ve_x_before1_bar
        self.ve_x_after1_bar = transverse_average(self.ve_x_after1)
        self.ve_x_after1_delta = self.ve_x_after1 - self.ve_x_after1_bar

        self.Pe_xx_bar = transverse_average(self.Pe_xx)
        self.Pe_xx_delta = self.Pe_xx - self.Pe_xx_bar
        self.Pe_xy_bar = transverse_average(self.Pe_xy)
        self.Pe_xy_delta = self.Pe_xy - self.Pe_xy_bar

        self.ne_bar = transverse_average(self.ne)
        self.ne_delta = self.ne - self.ne_bar

        self.A_e = - self.dvdt_e - self.vdvdx_e -(self.dPdx_e)/self.ne - (self.vyBz_e - self.vzBy_e)
        self.A_e_bar = transverse_average(self.A_e)
        self.A_e_delta = self.A_e - self.A_e_bar
        
        # ----------------------------------------------- POSITRONS -----------------------------------------------
        self.vp_x_bar = transverse_average(self.vp_x)
        self.vp_x_delta = self.vp_x - self.vp_x_bar
        self.vp_y_bar = transverse_average(self.vp_y)
        self.vp_y_delta = self.vp_y - self.vp_y_bar
        self.vp_z_bar = transverse_average(self.vp_z)
        self.vp_z_delta = self.vp_z - self.vp_z_bar

        self.vp_x_before1_bar = transverse_average(self.vp_x_before1)
        self.vp_x_before1_delta = self.vp_x_before1 - self.vp_x_before1_bar
        self.vp_x_after1_bar = transverse_average(self.vp_x_after1)
        self.vp_x_after1_delta = self.vp_x_after1 - self.vp_x_after1_bar

        self.Pp_xx_bar = transverse_average(self.Pp_xx)
        self.Pp_xx_delta = self.Pp_xx - self.Pp_xx_bar
        self.Pp_xy_bar = transverse_average(self.Pp_xy)
        self.Pp_xy_delta = self.Pp_xy - self.Pp_xy_bar

        self.n_p_bar = transverse_average(self.n_p)
        self.n_p_delta = self.n_p - self.n_p_bar

        self.A_p = self.dvdt_p + self.vdvdx_p  + (self.dPdx_p)/self.n_p - (self.vyBz_p - self.vzBy_p)
        self.A_p_bar = transverse_average(self.A_p)
        self.A_p_delta = self.A_p - self.A_p_bar
        
    def MeanFieldTheory_terms(self, species, get_terms=False):
        if species == 'electrons':
            self.dvdt_e_bar = (self.ve_x_after1_bar - self.ve_x_before1_bar) / (2*self.dt)
            self.vdvdx_e_bar = self.ve_x_bar * np.gradient(self.ve_x_bar, self.dx, axis=0)
            self.dPdx_e_bar = np.gradient(self.Pe_xx_bar, self.dx, axis=0)
            # dPdy_e_bar = np.gradient(Pe_xy_bar, dy)
            self.vyBz_e_bar = self.ve_y_bar*self.Bz_bar
            self.vzBy_e_bar = self.ve_z_bar*self.By_bar
            
            self.term1_e = transverse_average(self.A_e_delta * (self.ne_delta/self.ne_bar))
            self.term2_e = transverse_average((self.ne_delta/self.ne_bar) * (self.ve_x_after1_delta - self.ve_x_before1_delta) / (2*self.dt))
            self.term3_e = transverse_average((self.ne_delta/self.ne_bar) * self.ve_x_bar * np.gradient(self.ve_x_delta, self.dx, axis=1))
            self.term4_e = transverse_average((self.ne_delta/self.ne_bar) * self.ve_x_delta * np.gradient(self.ve_x_bar, self.dx, axis=0))

            self.term5_e = transverse_average(self.ve_x_delta * np.gradient(self.ve_x_delta, self.dx, axis=1))
            self.term6_e = transverse_average((self.ne_delta/self.ne_bar) * self.ve_x_delta * np.gradient(self.ve_x_delta, self.dx, axis=1))
            self.term7_e = transverse_average((1/self.ne_bar) * np.gradient(self.Pe_xx_delta, self.dx, axis=1))
            # self.term8_e = transverse_average(1/ne_bar * np.gradient(Pe_xy_delta, dy))

            self.term9_e = transverse_average(self.ve_y_delta*self.Bz_delta - self.ve_z_delta*self.By_delta)
            self.term10_e = transverse_average((self.ne_delta/self.ne_bar) * (self.ve_y_bar*self.Bz_delta - self.ve_z_bar*self.By_delta))
            self.term11_e = transverse_average((self.ne_delta/self.ne_bar) * (self.ve_y_delta*self.Bz_bar - self.ve_z_delta*self.By_bar))
            self.term12_e = transverse_average((self.ne_delta/self.ne_bar) * (self.ve_y_delta*self.Bz_delta - self.ve_z_delta*self.By_bar))
            
            if get_terms:
                dataframe_MFT_e = pd.DataFrame({'dvdt_e_bar': self.dvdt_e_bar, 'vdvdx_e_bar': self.vdvdx_e_bar, 'dPdx_e_bar': self.dPdx_e_bar, 
                                                'vyBz_e_bar': self.vyBz_e_bar, 'vzBy_e_bar': self.vzBy_e_bar, 'term1_e': self.term1_e, 'term2_e': self.term2_e,
                                                'term3_e': self.term3_e, 'term4_e': self.term4_e, 'term5_e': self.term5_e, 'term6_e': self.term6_e,
                                                'term7_e': self.term7_e, 'term9_e': self.term9_e, 'term10_e': self.term10_e, 'term11_e': self.term11_e,
                                                'term12_e': self.term12_e})
                return dataframe_MFT_e
            
        elif species == 'positrons':
            self.dvdt_p_bar = (self.vp_x_after1_bar - self.vp_x_before1_bar) / (2*self.dt)
            self.vdvdx_p_bar = self.vp_x_bar * np.gradient(self.vp_x_bar, self.dx, axis=0)
            self.dPdx_p_bar = np.gradient(self.Pp_xx_bar, self.dx, axis=0)
            # dPdy_p_bar = np.gradient(Pp_xy_bar, dy)
            self.vyBz_p_bar = self.vp_y_bar*self.Bz_bar
            self.vzBy_p_bar = self.vp_z_bar*self.By_bar
            
            self.term1_p = transverse_average(self.A_p_delta * (self.n_p_delta/self.n_p_bar))
            self.term2_p = transverse_average((self.n_p_delta/self.n_p_bar) * (self.vp_x_after1_delta - self.vp_x_before1_delta) / (2*self.dt))
            self.term3_p = transverse_average((self.n_p_delta/self.n_p_bar) * self.vp_x_bar * np.gradient(self.vp_x_delta, self.dx, axis=1))
            self.term4_p = transverse_average((self.n_p_delta/self.n_p_bar) * self.vp_x_delta * np.gradient(self.vp_x_bar, self.dx, axis=0))

            self.term5_p = transverse_average(self.vp_x_delta * np.gradient(self.vp_x_delta, self.dx, axis=1))
            self.term6_p = transverse_average((self.n_p_delta/self.n_p_bar) * self.vp_x_delta * np.gradient(self.vp_x_delta, self.dx, axis=1))
            self.term7_p = transverse_average((1/self.n_p_bar) * np.gradient(self.Pp_xx_delta, self.dx, axis=1))
            # term8_p = transverse_average(1/n_p_bar * np.gradient(Pp_xy_delta, dy))

            self.term9_p = transverse_average(self.vp_y_delta*self.Bz_delta - self.vp_z_delta*self.By_delta)
            self.term10_p = transverse_average((self.n_p_delta/self.n_p_bar) * (self.vp_y_bar*self.Bz_delta - self.vp_z_bar*self.By_delta))
            self.term11_p = transverse_average((self.n_p_delta/self.n_p_bar) * (self.vp_y_delta*self.Bz_bar - self.vp_z_delta*self.By_bar))
            self.term12_p = transverse_average((self.n_p_delta/self.n_p_bar) * (self.vp_y_delta*self.Bz_delta - self.vp_z_delta*self.By_bar))
            
            if get_terms:
                dataframe_MFT_p = pd.DataFrame({'dvdt_p_bar': self.dvdt_p_bar, 'vdvdx_p_bar': self.vdvdx_p_bar, 'dPdx_p_bar': self.dPdx_p_bar, 
                                                'vyBz_p_bar': self.vyBz_p_bar, 'vzBy_p_bar': self.vzBy_p_bar, 'term1_p': self.term1_p, 'term2_p': self.term2_p,
                                                'term3_p': self.term3_p, 'term4_p': self.term4_p, 'term5_p': self.term5_p, 'term6_p': self.term6_p,
                                                'term7_p': self.term7_p, 'term9_p': self.term9_p, 'term10_p': self.term10_p, 'term11_p': self.term11_p,
                                                'term12_p': self.term12_p})
                return dataframe_MFT_p
    
    def MeanFieldTheory(self, species):
        if species == 'electrons':
            self.MeanFieldTheory_terms(species)
            MFT_LHS_e_ = self.A_e_bar + self.dvdt_e_bar + self.vdvdx_e_bar + self.dPdx_e_bar/self.ne_bar + (self.vyBz_e_bar - self.vzBy_e_bar)
            MFT_LHS_e = integrate(MFT_LHS_e_, self.dx)
            MFT_RHS_e_ = -(self.term1_e + self.term2_e + self.term3_e + self.term4_e + self.term5_e + self.term6_e + self.term7_e + self.term9_e + self.term10_e + self.term11_e + self.term12_e)
            MFT_RHS_e = integrate(MFT_RHS_e_, self.dx)
            
            return MFT_LHS_e, MFT_RHS_e
        
        elif species == 'positrons':
            self.MeanFieldTheory_terms(species)
            MFT_LHS_p_ = self.A_p_bar - self.dvdt_p_bar - self.vdvdx_p_bar - self.dPdx_p_bar/self.n_p_bar + (self.vyBz_p_bar - self.vzBy_p_bar)
            MFT_LHS_p = integrate(MFT_LHS_p_, self.dx)
            MFT_RHS_p_ = (-self.term1_p + self.term2_p + self.term3_p + self.term4_p + self.term5_p + self.term6_p + self.term7_p - self.term9_p - self.term10_p - self.term11_p - self.term12_p)
            MFT_RHS_p = integrate(MFT_RHS_p_, self.dx)
            
            return MFT_LHS_p, MFT_RHS_p