#include <iostream>
#include <cmath>
#include <TMath.h>
#include <TRandom3.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TF1.h>

const int npoints = 12;
const int NE = 2000;
const double xmin = 1.0;
const double xmax = 20.0;
const double true_a = 0.5;
const double true_b = 1.3;
const double true_c = 0.5;
const double sigma = 0.2;
const int ndf = npoints - 3;

TF1* model = nullptr;
TH1D *hA = nullptr, *hB = nullptr, *hC = nullptr, *hChi2 = nullptr;

void fitPseudoExperiments()
{
    TRandom3 rand;
    double x[npoints], y[npoints], ey[npoints];
    double a_vals[NE], b_vals[NE], c_vals[NE], chi2_vals[NE];

    for (int ie = 0; ie < NE; ++ie)
    {
        // Generate x values
        for (int i = 0; i < npoints; ++i)
            x[i] = xmin + (xmax - xmin) * i / (npoints - 1);

        // Generate y values with noise
        for (int i = 0; i < npoints; ++i)
        {
            y[i] = true_a + true_b * TMath::Log(x[i]) + true_c * TMath::Power(TMath::Log(x[i]), 2) + rand.Gaus(0, sigma);
            ey[i] = sigma;
        }

        // Create TGraphErrors
        TGraphErrors* gr = new TGraphErrors(npoints, x, y, nullptr, ey);

        // Fit the model
        model = new TF1("model", "[0] + [1]*log(x) + [2]*pow(log(x),2)", xmin, xmax);
        gr->Fit(model, "Q");

        // Store fit parameters
        a_vals[ie] = model->GetParameter(0);
        b_vals[ie] = model->GetParameter(1);
        c_vals[ie] = model->GetParameter(2);

        // Calculate chi2
        chi2_vals[ie] = model->GetChisquare();

        // Fill histograms
        hA->Fill(a_vals[ie]);
        hB->Fill(b_vals[ie]);
        hC->Fill(c_vals[ie]);
        hChi2->Fill(chi2_vals[ie]);

        delete gr;
    }

    // Print summary statistics
    std::cout << "a: mean = " << hA->GetMean() << ", std = " << hA->GetStdDev() << std::endl;
    std::cout << "b: mean = " << hB->GetMean() << ", std = " << hB->GetStdDev() << std::endl;
    std::cout << "c: mean = " << hC->GetMean() << ", std = " << hC->GetStdDev() << std::endl;
    std::cout << "Chi2: mean = " << hChi2->GetMean() << ", std = " << hChi2->GetStdDev()
              << ", expected mean ≈ " << ndf << ", expected std ≈ " << std::sqrt(2 * ndf) << std::endl;
}

void createHistograms()
{
    hA = new TH1D("hA", "Distribution of a values", 50, 0.4, 0.6);
    hB = new TH1D("hB", "Distribution of b values", 50, 1.2, 1.4);
    hC = new TH1D("hC", "Distribution of c values", 50, 0.4, 0.6);
    hChi2 = new TH1D("hChi2", "Distribution of Chi2 values", 50, 0, 50);
}

void plotHistograms()
{
    TCanvas* c1 = new TCanvas("c1", "Fitted Parameters", 800, 600);
    c1->Divide(2, 2);

    c1->cd(1);
    hA->Draw();
    c1->cd(2);
    hB->Draw();
    c1->cd(3);
    hC->Draw();
    c1->cd(4);
    hChi2->Draw();

    c1->Update();
}

int main()
{
    createHistograms();
    fitPseudoExperiments();
    plotHistograms();

    // Clean up
    delete hA;
    delete hB;
    delete hC;
    delete hChi2;
    delete model;

    return 0;
}

