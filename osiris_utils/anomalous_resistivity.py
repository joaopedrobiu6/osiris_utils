import numpy as np
from .utils import *
from .data import *
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

class AnomalousResistivity:
    def __init__(self, quantity_folder, velocity_folder, quantity_iter, dump):
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
        
        ############################################################################################################
        self.dV1dt = (self.V1_a.data - self.V1_b.data)/(2*self.V1_a.dt)
        self.V1_dV1dx = self.V1.data * np.gradient(self.V1.data, self.V1.dx[0], axis=0)
        
        self.dT11nedx = np.gradient(self.T11.data*self.ne.data, self.T11.dx[0], axis=0)
        self.dT12nedy = np.gradient(self.T12.data*self.ne.data, self.T12.dx[1], axis=1)
        
        self.V2B3 = self.V2.data * self.B3.data
        self.V3B2 = self.V3.data * self.B2.data  
        
        self.E_vlasov = - self.dV1dt - self.V1_dV1dx - (1/self.ne.data)*(self.dT11nedx + self.dT12nedy) - (self.V2B3 - self.V3B2)

        ############################################################################################################
        
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

        ###########################################################################################################
        self.dV1dt_bar = (self.V1_a_bar - self.V1_b_bar)/(2*self.V1_a.dt)
        self.V1_dV1dx_bar = self.V1_bar*np.gradient(self.V1_bar, self.V1.dx[0], axis=0)
        
        self.dT11nedx_bar = np.gradient(self.T11_bar * self.ne_bar, self.T11.dx[0], axis=0)
        self.dT11nedx_delta = np.gradient(self.T11_delta * self.ne_delta, self.T11.dx[0], axis=0)
            
        self.V2V3_bar = self.V2_bar*self.B3_bar
        self.V3V2_bar = self.V3_bar*self.B2_bar
        ############################################################################################################
        
        self.momentum_bar = self.E_vlasov_bar + self.dV1dt_bar + self.V1_dV1dx_bar + self.dT11nedx_bar/self.ne_bar + self.V2V3_bar - self.V3V2_bar         
        self.momentum_bar = np.squeeze(self.momentum_bar)
        
        # convective term, u x B term
        self.term1 = transverse_average(self.V1_delta*np.gradient(self.V1_delta, self.V1.dx[0], axis=0))
        self.term2 = transverse_average(self.V2_delta*self.B3_delta)
        self.term3 = transverse_average(self.V3_delta*self.B2_delta)
        
        # thermal pressure gradient term xx
        self.term4 = transverse_average((np.gradient(self.ne_bar*self.T11_bar, self.T11.dx[0], axis=0)/self.ne.data)*(self.ne_delta/self.ne_bar))
        self.term5 = transverse_average(np.gradient(self.ne_bar*self.T11_delta, self.T11.dx[0], axis=0)/self.ne.data)
        self.term6 = transverse_average(np.gradient(self.ne_delta*self.T11_bar, self.T11.dx[0], axis=0)/self.ne.data)
        self.term7 = transverse_average(np.gradient(self.ne_delta*self.T11_delta, self.T11.dx[0], axis=0)/self.ne.data)

        # thermal pressure gradient term xy
        self.term8 = transverse_average(np.gradient(self.ne_bar*self.T12_delta, self.T12.dx[1], axis=1)/self.ne.data)
        self.term9 = transverse_average(np.gradient(self.ne_delta*self.T12_bar, self.T12.dx[1], axis=1)/self.ne.data)
        self.term10 = transverse_average(np.gradient(self.ne_delta*self.T12_delta, self.T12.dx[1], axis=1)/self.ne.data)
        

        self.eta =  - self.term1 - self.term2 + self.term3 + self.term4 - self.term5 - self.term6 - self.term7 - self.term8 - self.term9 - self.term10        
        self.eta_dominant = - self.term2 + self.term4 - self.term5 - self.term6 - self.term7 - self.term8 - self.term10

    def Momentum(self):
        return self.momentum_bar, self.eta, self.eta_dominant
    
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