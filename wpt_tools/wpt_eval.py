# VNA WPT measurement toolchain for PicoVNA 106
# Takuya Sasatani

# Software
# - Windows 11
# - Anaconda 3 (Python 3.9.7)
# - PicoVNA 3 software and drivers need to be installed

# Hardware
# - PicoVNA 106
# - Calibration kit (Calibration file needs to be obtained using PicoVNA 3 software before measurement)

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

import skrf as rf
from pylab import *
from skrf import Network, Frequency
import math

import sys

from scipy.optimize import curve_fit
from scipy.optimize import fmin
import scipy.optimize as optimize
import sklearn.metrics as metrics


class wpt_eval:
    def __init__(self):
        self.nw = None
        self.f_narrow_index_start = None
        self.f_narrow_index_stop = None
        self.target_f_index = None
        self.sweeppoint = None
        self.range_f = None
        self.target_f = None

    def import_touchstone(self, filename = ''):
        if filename == '':
            root = tk.Tk()
            root.withdraw()
            root.update()
            filename = filedialog.askopenfilename(initialdir="./")
            root.quit()

        
        self.nw = rf.Network(filename)
        self.sweeppoint = np.size(self.nw.frequency.f)
        print(filename)

    def export_touchstone(self, filename):
        self.nw.write_touchstone(filename)

    def picoVNA_measure(self, cal_file, start_f, end_f, sweep_points, power_level, RBW, z0, progid):
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

    # Narrow down frequency range and find index of target frequency
    def set_f_target_range(self, target_f, range_f):
        self.target_f = target_f
        self.range_f = range_f

        self.f_narrow_index_start = self.sweeppoint
        self.f_narrow_index_stop = 0

        d_target_f = self.range_f

        for f_index in range(self.sweeppoint):
            if abs(target_f - self.nw.frequency.f[f_index]) < range_f/2:
                if self.f_narrow_index_start > f_index:
                    self.f_narrow_index_start = f_index
                if self.f_narrow_index_stop < f_index:
                    self.f_narrow_index_stop = f_index

                f_temp = self.nw.frequency.f[f_index]

                if abs(target_f - f_temp) < d_target_f:
                    d_target_f = abs(target_f - f_temp)
                    self.target_f_index = f_index

    # Efficiency and optimal load analysis (for general 2-port networks)

    # Reference: Y. Narusue, et al., "Load optimization factors for analyzing the efficiency of wireless power transfer systems using two-port network parameters," IEICE ELEX, 2020.
    # Unstable when far from resonant frequency (probably because to S to Z conversion becomes unstable)

    def efficiency_load_analysis(self, rx_port=2, show_plot=1, show_data=1):
        f_plot = []
        r_det = []
        kq2 = []
        r_opt = []
        x_opt = []
        eff_opt = []

        max_eff_opt = 0
        max_f_index = None

        if self.target_f == None:
            print('execute set_f_target_range() before this operation')
            sys.exit()

        for f_index in range(self.sweeppoint):
            if abs(self.target_f - self.nw.frequency.f[f_index]) < self.range_f/2:
                if rx_port == 2:
                    Z11 = self.nw.z[f_index, 0, 0]
                    Z22 = self.nw.z[f_index, 1, 1]
                elif rx_port == 1:
                    Z11 = self.nw.z[f_index, 1, 1]
                    Z22 = self.nw.z[f_index, 0, 0]
                else:
                    print('set rx_port to 1 or 2.')
                    sys.exit()
                Zm = self.nw.z[f_index, 0, 1]
                f_temp = self.nw.frequency.f[f_index]
                r_det_temp = Z11.real * Z22.real - Zm.real**2

                kq2_temp = (Zm.real**2 + Zm.imag**2)/r_det_temp
                r_opt_temp = r_det_temp / Z11.real * np.sqrt(1 + kq2_temp)
                x_opt_temp = Zm.real * Zm.imag - Z11.real * Z22.imag / Z11.real
                eff_opt_temp = kq2_temp / (1+np.sqrt(1 + kq2_temp))**2

                f_plot.append(f_temp)
                r_det.append(r_det_temp)
                kq2.append(kq2_temp)
                r_opt.append(r_opt_temp)
                x_opt.append(x_opt_temp)
                eff_opt.append(eff_opt_temp)

                if max_eff_opt < eff_opt_temp:
                    max_f_index = f_index
        if show_plot == 1:
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))

            axs[0].plot(f_plot, eff_opt)
            axs[0].set_title("Maximum efficiency")
            axs[0].set_xlabel("Frequency")
            axs[0].set_ylabel("Efficiency")
            axs[0].axvline(self.target_f, color='gray', lw=1)

            axs[1].plot(f_plot, r_opt)
            axs[1].set_title("Optimum Re($Z_\mathrm{load}$)")
            axs[1].set_xlabel("Frequency")
            axs[1].set_ylabel("Optimum Re($Z_\mathrm{load}$) ($\Omega$)")
            axs[1].axvline(self.target_f, color='gray', lw=1)

            axs[2].plot(f_plot, x_opt)
            axs[2].set_title("Optimum Im($Z_\mathrm{load}$)")
            axs[2].set_xlabel("Frequency")
            axs[2].set_ylabel("Optimum Im($Z_\mathrm{load}$) ($\Omega$)")
            axs[2].axvline(self.target_f, color='gray', lw=1)

            fig.tight_layout()

        max_f_plot = f_plot[max_f_index]
        max_eff_opt = eff_opt[max_f_index]
        max_r_opt = r_opt[max_f_index]
        max_x_opt = x_opt[max_f_index]

        if show_data == 1:
            print('Target frequency: %.3e' % (max_f_plot))
            print('Maximum efficiency: %.2f' % (max_eff_opt))
            print('Optimum Re(Zload): %.2f' % (max_r_opt))
            print('Optimum Im(Zload): %.2f' % (max_x_opt))

        return max_f_plot, max_eff_opt, max_r_opt, max_x_opt

    # Plot Z-parameters (full-range)
    def plot_z_full(self):
        fig, axs = plt.subplots(1, 4, figsize=(18, 3.5))
        twin = ["init"] * 4
        pr = ["init"] * 4
        pi = ["init"] * 4

        for rx_port in range(1, 3):
            for tx_port in range(1, 3):
                plot_index = (rx_port-1)*2+(tx_port-1)*1
                axs[plot_index].set_title("Z"+str(rx_port)+str(tx_port))
                twin[plot_index] = axs[plot_index].twinx()
                pr[plot_index], = axs[plot_index].plot(
                    self.nw.frequency.f, self.nw.z[:, rx_port-1, tx_port-1].real, label="real(z)")
                pi[plot_index], = twin[plot_index].plot(
                    self.nw.frequency.f, self.nw.z[:, rx_port-1, tx_port-1].imag, "r-", label="imag(z)")
                axs[plot_index].set_xlabel("frequency")
                axs[plot_index].set_ylabel(
                    "re(Z"+str(rx_port)+str(tx_port)+") ($\Omega$)")
                twin[plot_index].set_ylabel(
                    "im(Z"+str(rx_port)+str(tx_port)+") ($\Omega$)")
                axs[plot_index].yaxis.label.set_color(
                    pr[plot_index].get_color())
                twin[plot_index].yaxis.label.set_color(
                    pi[plot_index].get_color())
                if self.target_f != None:
                    axs[plot_index].axvline(self.target_f, color='gray', lw=1)
                axs[plot_index].set_ylim(
                    (-abs(self.nw.z[:, rx_port-1, tx_port-1].real).max(), abs(self.nw.z[:, rx_port-1, tx_port-1].real).max()))
                twin[plot_index].set_ylim(
                    (-abs(self.nw.z[:, rx_port-1, tx_port-1].imag).max(), abs(self.nw.z[:, rx_port-1, tx_port-1].imag).max()))
                axs[plot_index].axhline(0, color='gray', lw=1)
        fig.tight_layout()

    # Curve-fitting and Z-matrix plot (narrow-range)
    def plot_z_narrow_fit(self, show_plot = 1, show_fit = 1):
        if self.target_f == None:
            print('execute set_f_target_range() before this operation')
            sys.exit()

        def series_lcr_xself(x, ls, cs):
            return 2*math.pi*x*ls - 1/(2*math.pi*x*cs)

        def series_lcr_rself(x, r):
            return 0*x + r

        def series_lcr_xm(x, lm):
            return 2*math.pi*x*lm

        popt, _ = curve_fit(series_lcr_xself, self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop],
                            self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].imag, p0=np.asarray([1e-6, 1e-9]), maxfev=10000)
        ls1, cs1 = popt
        r2 = metrics.r2_score(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].imag, series_lcr_xself(
            self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], ls1, cs1))
        if show_fit == 1: print('R2 for fitting Ls1, Cs1: %f' % (r2))

        popt, _ = curve_fit(series_lcr_rself, self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop],
                            self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].real, p0=np.asarray([1]), maxfev=10000)
        rs1 = popt

        if self.nw.nports == 2:
            popt, _ = curve_fit(series_lcr_xself, self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop],
                                self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 1, 1].imag, p0=np.asarray([1e-6, 1e-9]), maxfev=10000)
            ls2, cs2 = popt

            r2 = metrics.r2_score(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 1, 1].imag, series_lcr_xself(
                self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], ls2, cs2))
            if show_fit == 1: print('R2 for fitting Ls2, Cs2: %f' % (r2))

            popt, _ = curve_fit(series_lcr_rself, self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop],
                                self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 1, 1].real, p0=np.asarray([1]), maxfev=10000)
            rs2 = popt

            popt, _ = curve_fit(series_lcr_xm, self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop],
                                self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 1].imag, p0=np.asarray([1e-6]), maxfev=10000)
            lm = popt
            r2 = metrics.r2_score(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 1].imag, series_lcr_xm(
                self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], lm))
            #print('R2 for fitting Lm: %f' % (r2))
        
        if show_fit == 1: 
            print('Self impedance at target frequency\n')
            print('Re(Z11): %.2e\nIm(Z11): %.2e\n' % (self.nw.z[self.target_f_index, 0, 0].real, self.nw.z[self.target_f_index, 0, 0].imag))

            if self.nw.nports == 2:
                print('Re(Z22): %.2e\nIm(Z22) %.2e\n' % (self.nw.z[self.target_f_index, 1, 1].real, self.nw.z[self.target_f_index, 1, 1].imag))

            print('Fitting values assuming a pair of series LCR resonators\n')
            print('Ls1: %.2e, Cs1: %.2e, Rs1: %.2e, f_1: %.3e, Q_1 (approximate, @%.3e Hz): %.2e' %
                (ls1, cs1, rs1, 1/(2*np.pi*np.sqrt(ls1*cs1)), self.target_f, 2*np.pi*self.target_f*ls1/rs1))
            if self.nw.nports == 2:
                print('Ls2: %.2e, Cs2: %.2e, Rs2: %.2e, f_2: %.3e, Q_2 (approximate, @%.3e Hz): %.2e' %
                (ls2, cs2, rs2, 1/(2*np.pi*np.sqrt(ls2*cs2)), self.target_f, 2*np.pi*self.target_f*ls2/rs2))
                print('Lm: %.2e, km: %.3f' % (lm, lm/np.sqrt(ls1*ls2)))
        
        if show_plot == 1:
            if self.nw.nports == 1:
                fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))
                twin = ["init"] * 1
                pr = ["init"] * 1
                pi = ["init"] * 1

                axs.set_title("Z11")
                twin = axs.twinx()
                pr[0], = axs.plot(
                    self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].real, label="real(z)", lw=3)
                pi[0], = twin.plot(
                    self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].imag, "r-", label="imag(z)", lw=3)
                axs.set_xlabel("frequency")
                axs.set_ylabel("re(Z11) ($\Omega$)")
                twin.set_ylabel("im(Z) ($\Omega$)")
                axs.yaxis.label.set_color(pr[0].get_color())
                twin.yaxis.label.set_color(pi[0].get_color())
                axs.axvline(self.target_f, color='gray', lw=1)
                axs.set_ylim((-1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].real).max(
                    ), 1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].real).max()))
                twin.set_ylim((-1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].imag).max(
                    ), 1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, 0, 0].imag).max()))
                axs.axhline(0, color='gray', lw=1)

                twin.plot(self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], series_lcr_xself(
                    self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], ls1, cs1), label="imag(z) fitting", color='green')
                fig.tight_layout()

            if self.nw.nports == 2:
                fig, axs = plt.subplots(1, 4, figsize=(18, 3.5))
                twin = ["init"] * 4
                pr = ["init"] * 4
                pi = ["init"] * 4

                for rx_port in range(1, 3):
                    for tx_port in range(1, 3):
                        plot_index = (rx_port-1)*2+(tx_port-1)*1
                        axs[plot_index].set_title("Z"+str(rx_port)+str(tx_port))
                        twin[plot_index] = axs[plot_index].twinx()
                        pr[plot_index], = axs[plot_index].plot(
                            self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, rx_port-1, tx_port-1].real, label="real(z)", lw=3)
                        pi[plot_index], = twin[plot_index].plot(
                            self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, rx_port-1, tx_port-1].imag, "r-", label="imag(z)", lw=3)
                        axs[plot_index].set_xlabel("frequency")
                        axs[plot_index].set_ylabel(
                            "re(Z"+str(rx_port)+str(tx_port)+") ($\Omega$)")
                        twin[plot_index].set_ylabel(
                            "im(Z"+str(rx_port)+str(tx_port)+") ($\Omega$)")
                        axs[plot_index].yaxis.label.set_color(
                            pr[plot_index].get_color())
                        twin[plot_index].yaxis.label.set_color(
                            pi[plot_index].get_color())
                        axs[plot_index].axvline(self.target_f, color='gray', lw=1)
                        axs[plot_index].set_ylim((-1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, rx_port-1, tx_port-1].real).max(
                        ), 1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, rx_port-1, tx_port-1].real).max()))
                        twin[plot_index].set_ylim((-1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, rx_port-1, tx_port-1].imag).max(
                        ), 1.5*abs(self.nw.z[self.f_narrow_index_start:self.f_narrow_index_stop, rx_port-1, tx_port-1].imag).max()))
                        axs[plot_index].axhline(0, color='gray', lw=1)

                twin[0].plot(self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], series_lcr_xself(
                    self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], ls1, cs1), label="imag(z) fitting", color='green')
                twin[3].plot(self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], series_lcr_xself(
                    self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], ls2, cs2), label="imag(z) fitting", color='green')
                twin[1].plot(self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], series_lcr_xm(
                    self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], lm), label="imag(z) fitting", color='green')
                twin[2].plot(self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], series_lcr_xm(
                    self.nw.frequency.f[self.f_narrow_index_start:self.f_narrow_index_stop], lm), label="imag(z) fitting", color='green')

                fig.tight_layout()

        return ls1, cs1, rs1, ls2, cs2, rs2, lm

    def rxc_filter_calc(self, rx_port, rload, c_network = 'CpCsRl'):
        ls1, _, rs1, ls2, _, rs2, _ = self.plot_z_narrow_fit(show_plot=0, show_fit=0)
        max_f_plot, max_eff_opt, max_r_opt, max_x_opt = self.efficiency_load_analysis(rx_port=rx_port, show_plot=0, show_data=0)

        max_w_plot = 2 * np.pi * max_f_plot
        if rx_port == 1:
            lrx = ls1
        elif rx_port == 2:
            lrx = ls2
        else:
            print('set rx_port parameter to 1 or 2')
            sys.exit()

        print('Target frequency: %.3e' % (max_f_plot))
        print('Maximum efficiency: %.2f' % (max_eff_opt))
        print('Receiver inductance: %.2e' % (lrx))
        print('Optimum load: %.2f' % (max_r_opt))
        print('Target Rload: %.2f\n' % (rload))
        

        if c_network == 'CpCsRl':
            def Z(params):
                cp, cs = params
                return 1/((1j * max_w_plot * cp) + 1/((1/(1j*max_w_plot*cs)+rload))) + 1j*max_w_plot*lrx
            def Zerror(params):
                return np.linalg.norm([Z(params).real-max_r_opt, Z(params).imag])
        sol = fmin(Zerror, np.array([100e-12,100e-12]),xtol=1e-9, ftol=1e-9)
        print(sol)

    def optimal_load_plot(self, min_rez, min_imz, max_rez, max_imz, step_rez, step_imz, input_voltage, rx_port=2):
    # Optimal load visualization
    # Imura, "Wireless Power Transfer: Using Magnetic and Electric Resonance Coupling Techniques," Springer Singapore 2020.
        rez_list = np.arange(min_rez, max_rez, step_rez)
        imz_list = np.arange(min_imz, max_imz, step_imz)
        eff_grid = np.zeros((rez_list.size, imz_list.size))
        Pin = np.zeros((rez_list.size, imz_list.size))
        Pout = np.zeros((rez_list.size, imz_list.size))

        if rx_port == 2:
            Z11 = self.nw.z[self.target_f_index, 0, 0]
            Z22 = self.nw.z[self.target_f_index, 1, 1]
        elif rx_port == 1:
            Z11 = self.nw.z[self.target_f_index, 1, 1]
            Z22 = self.nw.z[self.target_f_index, 0, 0]
        else:
            print('set rx_port to 1 or 2.')
            sys.exit()
        
        Zm = self.nw.z[self.target_f_index, 0, 1]

        for rez_index in range(rez_list.size):
            for imz_index in range(imz_list.size):
                ZL = rez_list[rez_index] + 1j*imz_list[imz_index]
                V1 = input_voltage  # arbitrary
                I1 = (Z22+ZL)/(Z11*(Z22+ZL)-Zm**2)*V1
                I2 = -Zm/(Z11*(Z22+ZL)-Zm**2)*V1
                V2 = Zm*ZL/(Z11*(Z22+ZL)-Zm**2)*V1

                Pin[rez_index][imz_index] = (V1*I1.conjugate()).real
                Pout[rez_index][imz_index] = (V2*(-I2.conjugate())).real
                eff_grid[rez_index][imz_index] = (
                    V2*(-I2.conjugate())).real/(V1*I1.conjugate()).real

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        c = axs[0].pcolor(imz_list, rez_list, eff_grid,
                          cmap='hot', vmin=0, vmax=1, shading='auto')
        fig.colorbar(c, ax=axs[0])
        axs[0].set_title(
            'Efficiency @ '+format(self.nw.frequency.f[self.target_f_index], '3.2e')+' Hz')
        axs[0].set_ylabel('Re($Z_{\mathrm{load}}$)')
        axs[0].set_xlabel('Im($Z_{\mathrm{load}}$)')

        c = axs[1].pcolor(imz_list, rez_list, Pin, cmap='hot',
                          vmin=0, vmax=Pin.max(), shading='auto')
        fig.colorbar(c, ax=axs[1])
        axs[1].set_title('Input Power (W) @ ' +
                         format(self.nw.frequency.f[self.target_f_index], '3.2e')+' Hz')
        axs[1].set_ylabel('Re($Z_{\mathrm{load}}$)')
        axs[1].set_xlabel('Im($Z_{\mathrm{load}}$)')

        c = axs[2].pcolor(imz_list, rez_list, Pout, cmap='hot',
                          vmin=0, vmax=Pin.max(), shading='auto')
        fig.colorbar(c, ax=axs[2])
        axs[2].set_title('Output Power (W) @ ' +
                         format(self.nw.frequency.f[self.target_f_index], '3.2e')+' Hz')
        axs[2].set_ylabel('Re($Z_{\mathrm{load}}$)')
        axs[2].set_xlabel('Im($Z_{\mathrm{load}}$)')

        fig.tight_layout()
