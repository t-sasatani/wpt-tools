"""
Instrument control. Depricated for now.
"""
class picoVNA_manager:
    """
    Class for handling picoVNA measurement. Depricated
    """
    def picoVNA_measure(self, cal_file, start_f, end_f, sweep_points, power_level, RBW, z0, progid):
        """
        picoVNA control. Depricated for now.
        """
        import win32com.client
        picoVNACOMObj = win32com.client.Dispatch(progid)

        # Connect PicoVNA
        # 0 probably means no VNAs are found
        print("Connecting VNA")
        findVNA = picoVNACOMObj.FND()
        print('VNA #' + str(findVNA) + ' Loaded')

        # Set frequency plan
        # parameters need to be same as calibration file
        print("Set frequency plan")
        step_f = (end_f - start_f)/(sweep_points - 1)

        ans_freq_plan = picoVNACOMObj.SetFreqPlan(
            start_f/1e6, step_f/1e6, sweep_points, power_level, RBW)
        print("Result " + str(ans_freq_plan))

        # Load calibration file and measure
        print("Load Calibration")
        ans_calibration = picoVNACOMObj.LoadCal(cal_file)
        print("Result " + str(ans_calibration))

        print("Making Measurement")
        picoVNACOMObj.Measure('ALL')

        print("getting full S-matrix (RI form)")

        spar_list = {}
        for rx_port in range(1, 3):
            for tx_port in range(1, 3):
                for ri in ["real", "imag"]:
                    temp_spar = picoVNACOMObj.GetData(
                        "S"+str(rx_port)+str(tx_port), ri, 0)
                    spar_list["s{0}{1}_{2}".format(rx_port, tx_port, ri)] = np.array(
                        temp_spar.split(',')).astype(float)

        # Convert PicoVNA output to Scikit.rf NW object
        f = spar_list['s11_real'][0::2]
        s11 = spar_list['s11_real'][1::2] + 1j*spar_list['s11_imag'][1::2]
        s21 = spar_list['s21_real'][1::2] + 1j*spar_list['s21_imag'][1::2]
        s12 = spar_list['s12_real'][1::2] + 1j*spar_list['s12_imag'][1::2]
        s22 = spar_list['s22_real'][1::2] + 1j*spar_list['s22_imag'][1::2]

        # define a 2x2 s-matrix at a given frequency
        def si(i):
            ''' s-matrix at frequency i'''
            return np.array(([s11[i], s12[i]], [s21[i], s22[i]]), dtype='complex')

        # number of frequency points
        self.sweeppoint = np.size(f)

        # stack matrices along 3rd axis to create (2x2xN) array
        s = si(0)
        for i in range(1, self.sweeppoint):
            s = np.dstack([s, si(i)])

        # re-shape into (Nx2x2)
        s = np.swapaxes(s, 0, 2)

        f_obj = rf.Frequency.from_f(f, unit='hz')

        # create network object with frequency converted to GHz units
        self.nw = rf.Network(name='nw_from_numpy', s=s, frequency=f_obj, z0=z0)
        print(self)

        picoVNACOMObj.CloseVNA()
        print("VNA Closed")

    def export_touchstone(self, filename):
        self.nw.write_touchstone(filename)

