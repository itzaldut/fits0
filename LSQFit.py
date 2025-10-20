import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# --- parameters for pseudo‑experiments and model ---
true_a = 0.5
true_b = 1.3
true_c = 0.5

xmin = 1.0
xmax = 20.0
npoints = 12        
sigma = 0.2         
NE = 2000           # number of pseudo‑experiments to run

# define model function
def model(x, a, b, c):
    return a + b * np.log(x) + c * (np.log(x))**2

# storage arrays
a_vals = np.zeros(NE)
b_vals = np.zeros(NE)
c_vals = np.zeros(NE)
chi2_vals = np.zeros(NE)
red_chi2_vals = np.zeros(NE)

# degrees of freedom
ndf = npoints - 3

# x grid
x = np.linspace(xmin, xmax, npoints)

for ie in range(NE):
    # generate y data + uncertainties
    y = model(x, true_a, true_b, true_c) + np.random.normal(loc=0.0, scale=sigma, size=npoints)
    ey = np.full_like(y, fill_value=sigma)
    # perform fit
    popt, pcov = curve_fit(model, x, y, p0=[true_a, true_b, true_c], sigma=ey, absolute_sigma=True)
    a_fit, b_fit, c_fit = popt
    # compute residuals and chi²
    y_fit = model(x, a_fit, b_fit, c_fit)
    resid = (y - y_fit) / ey
    chi2 = np.sum(resid**2)
    # store
    a_vals[ie] = a_fit
    b_vals[ie] = b_fit
    c_vals[ie] = c_fit
    chi2_vals[ie] = chi2
    red_chi2_vals[ie] = chi2 / ndf

# Create the PDF and save plots
with PdfPages('LSQFit.pdf') as pdf:

    # --- First panel: histograms of a, b, c, chi² ---
    fig1, axes1 = plt.subplots(2,2, figsize=(10,8))
    axes1 = axes1.flatten()
    axes1[0].hist(a_vals, bins=50, alpha=0.8)
    axes1[0].set_xlabel('a fit results')
    axes1[0].set_ylabel('Counts')
    axes1[1].hist(b_vals, bins=50, alpha=0.8)
    axes1[1].set_xlabel('b fit results')
    axes1[1].set_ylabel('Counts')
    axes1[2].hist(c_vals, bins=50, alpha=0.8)
    axes1[2].set_xlabel('c fit results')
    axes1[2].set_ylabel('Counts')
    axes1[3].hist(chi2_vals, bins=50, alpha=0.8)
    axes1[3].set_xlabel('χ²')
    axes1[3].set_ylabel('Counts')
    fig1.suptitle('Distributions of fitted parameters and χ²')
    fig1.subplots_adjust(top=0.90)
    pdf.savefig(fig1)
    plt.close(fig1)

    # Print out summary statistics in console
    print("a: mean = {:.4f}, std = {:.4f}".format(np.mean(a_vals), np.std(a_vals)))
    print("b: mean = {:.4f}, std = {:.4f}".format(np.mean(b_vals), np.std(b_vals)))
    print("c: mean = {:.4f}, std = {:.4f}".format(np.mean(c_vals), np.std(c_vals)))
    print("χ²: mean = {:.2f}, std = {:.2f}, expected mean ≈ {}, expected σ ≈ {}".format(
        np.mean(chi2_vals), np.std(chi2_vals), ndf, np.sqrt(2*ndf)))

    # --- Second panel: parameter correlations and reduced χ² ---
    fig2, axes2 = plt.subplots(2,2, figsize=(10,8))
    axes2 = axes2.flatten()
    hb0 = axes2[0].hist2d(a_vals, b_vals, bins=50, cmap='viridis')
    axes2[0].set_xlabel('a fit')
    axes2[0].set_ylabel('b fit')
    fig2.colorbar(hb0[3], ax=axes2[0])
    hb1 = axes2[1].hist2d(a_vals, c_vals, bins=50, cmap='viridis')
    axes2[1].set_xlabel('a fit')
    axes2[1].set_ylabel('c fit')
    fig2.colorbar(hb1[3], ax=axes2[1])
    hb2 = axes2[2].hist2d(b_vals, c_vals, bins=50, cmap='viridis')
    axes2[2].set_xlabel('b fit')
    axes2[2].set_ylabel('c fit')
    fig2.colorbar(hb2[3], ax=axes2[2])
    axes2[3].hist(red_chi2_vals, bins=50, alpha=0.8)
    axes2[3].set_xlabel('reduced χ² (χ²/ndf)')
    axes2[3].set_ylabel('Counts')
    fig2.suptitle('Parameter correlations and reduced χ²')
    fig2.subplots_adjust(top=0.90)
    pdf.savefig(fig2)
    plt.close(fig2)

print("Saved plots to LSQFit.pdf")

