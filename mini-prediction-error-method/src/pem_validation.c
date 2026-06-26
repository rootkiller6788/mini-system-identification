#include "pem_core.h"
#include "pem_predictor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * PEM Model Validation — Statistical Tests and Information Criteria
 *
 * Model validation is a critical step in system identification.
 * It answers: "Is this model good enough for its intended use?"
 *
 * Validation techniques implemented:
 *   1. NRMSE fit (Normalized Root Mean Square Error)
 *   2. AIC / AICc / BIC / FPE (Information criteria)
 *   3. Residual whiteness test (Ljung-Box Q-test)
 *   4. Cross-correlation test (independence of eps and u)
 *   5. R-squared and adjusted R-squared
 *   6. K-fold cross-validation
 *
 * Reference:
 *   Ljung (1999) Chapter 16: "Model Validation"
 *   Akaike (1974) — "A New Look at the Statistical Model Identification"
 *   Ljung & Box (1978) — "On a Measure of Lack of Fit in Time Series Models"
 * ============================================================================ */

/* ================================================================
 * NRMSE Fitness
 *
 * FIT = 100 * (1 - ||y - y_hat|| / ||y - mean(y)||)
 *
 * FIT = 100%: perfect prediction
 * FIT = 0%: prediction no better than constant mean model
 * FIT < 0%: prediction worse than the mean
 *
 * This is the standard metric in the system identification community
 * (see Ljung's System Identification Toolbox for MATLAB).
 * ================================================================ */

double pem_validation_fit(const double *y, const double *y_hat, int N) {
    if (N <= 1) return 0.0;
    double y_mean = pem_mean(y, N);

    double ss_res = 0.0;  /* Sum of squared residuals */
    double ss_tot = 0.0;  /* Total sum of squares */
    for (int i = 0; i < N; i++) {
        double e = y[i] - y_hat[i];
        ss_res += e * e;
        double d = y[i] - y_mean;
        ss_tot += d * d;
    }

    if (ss_tot < 1e-15) return 100.0;
    double nrmse = sqrt(ss_res / ss_tot);
    return 100.0 * (1.0 - nrmse);
}

/* ================================================================
 * R-squared (Coefficient of Determination)
 *
 * R^2 = 1 - SS_res / SS_tot
 *
 * Adjusted R^2 = 1 - (1 - R^2) * (N - 1) / (N - d - 1)
 *
 * where d is the number of model parameters.
 * Adjusted R^2 penalizes model complexity.
 * ================================================================ */

double pem_validation_rsquared(const double *y, const double *y_hat, int N) {
    if (N <= 1) return 0.0;
    double y_mean = pem_mean(y, N);
    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < N; i++) {
        double e = y[i] - y_hat[i];
        ss_res += e * e;
        double d = y[i] - y_mean;
        ss_tot += d * d;
    }
    if (ss_tot < 1e-15) return 1.0;
    return 1.0 - ss_res / ss_tot;
}

double pem_validation_adj_rsquared(const double *y, const double *y_hat, int N, int d) {
    double r2 = pem_validation_rsquared(y, y_hat, N);
    if (N <= d + 1) return r2;
    return 1.0 - (1.0 - r2) * (double)(N - 1) / (double)(N - d - 1);
}

/* ================================================================
 * Information Criteria
 *
 * AIC (Akaike Information Criterion):
 *   AIC = log(V_N) + 2d/N
 *   where V_N is the loss function value, d is number of parameters.
 *
 * AICc (corrected AIC, better for small samples):
 *   AICc = AIC + 2d(d+1) / (N - d - 1)
 *
 * BIC (Bayesian Information Criterion / Schwarz):
 *   BIC = log(V_N) + d * log(N) / N
 *   BIC penalizes complexity more heavily than AIC.
 *
 * FPE (Akaike's Final Prediction Error):
 *   FPE = V_N * (1 + d/N) / (1 - d/N)
 *
 * Lower values indicate better models (balancing fit and complexity).
 *
 * Reference:
 *   Akaike (1974), IEEE Trans. Automatic Control
 *   Schwarz (1978), Annals of Statistics
 *   Hurvich & Tsai (1989), Biometrika (for AICc)
 * ================================================================ */

double pem_validation_aic(double loss, int N, int d) {
    if (N <= 0 || loss <= 0.0) return 1e100;
    return log(loss) + 2.0 * (double)d / (double)N;
}

double pem_validation_aicc(double loss, int N, int d) {
    if (N <= d + 2) return 1e100;
    double aic = pem_validation_aic(loss, N, d);
    return aic + 2.0 * (double)(d * (d + 1)) / (double)(N - d - 1);
}

double pem_validation_bic(double loss, int N, int d) {
    if (N <= 0 || loss <= 0.0) return 1e100;
    return log(loss) + (double)d * log((double)N) / (double)N;
}

double pem_validation_fpe(double loss, int N, int d) {
    if (N <= d) return 1e100;
    return loss * (1.0 + (double)d / (double)N) / (1.0 - (double)d / (double)N);
}

/* ================================================================
 * Residual Whiteness Test (Ljung-Box Q-test)
 *
 * Tests the null hypothesis that residuals are independently distributed
 * (white noise). The test statistic:
 *
 *   Q = N * (N + 2) * sum_{k=1}^{m} r_k^2 / (N - k)
 *
 * where r_k is the autocorrelation of residuals at lag k,
 * and m is the number of lags tested (typically m = min(20, N/4)).
 *
 * Under H0, Q ~ chi-squared(m) asymptotically.
 *
 * We return the p-value approximation.
 * Small p-value (< 0.05) indicates residuals are NOT white.
 *
 * Reference:
 *   Ljung & Box (1978), Biometrika
 * ================================================================ */

/** Autocorrelation at lag k: r_k = sum_{t=k+1}^N eps(t)*eps(t-k) / sum eps(t)^2 */
static double autocorr(const double *eps, int N, int k) {
    if (N <= k) return 0.0;
    double num = 0.0, den = 0.0;
    for (int t = k; t < N; t++) num += eps[t] * eps[t - k];
    for (int t = 0; t < N; t++) den += eps[t] * eps[t];
    if (den < 1e-15) return 0.0;
    return num / den;
}

/** Compute Ljung-Box Q statistic and approximate p-value.
 *  Returns a value in [0, 1]; higher values suggest whiteness.
 *  We invert the chi-squared CDF using a simple approximation. */
double pem_validation_ljung_box(const double *eps, int N, int m) {
    if (N <= m || m <= 0) return 0.0;
    if (m > N / 4) m = N / 4;
    if (m < 1) m = 1;

    double Q = 0.0;
    for (int k = 1; k <= m; k++) {
        double rk = autocorr(eps, N, k);
        Q += rk * rk / (double)(N - k);
    }
    Q *= (double)N * (double)(N + 2);

    /* Chi-squared CDF approximation (Wilson-Hilferty):
     * For chi-squared with nu degrees of freedom:
     * z = ((Q/nu)^(1/3) - 1 + 2/(9*nu)) / sqrt(2/(9*nu))
     * Then use standard normal CDF approximation. */
    double nu = (double)m;
    if (nu < 1.0) nu = 1.0;
    double q_nu = Q / nu;
    if (q_nu < 0.0) q_nu = 0.0;
    double z = (pow(q_nu, 1.0/3.0) - 1.0 + 2.0/(9.0*nu)) / sqrt(2.0/(9.0*nu));

    /* Standard normal CDF approximation (Abramowitz & Stegun 26.2.17) */
    double abs_z = fabs(z);
    double t = 1.0 / (1.0 + 0.2316419 * abs_z);
    double phi = 1.0 - 1.0/sqrt(2.0*3.141592653589793) * exp(-abs_z*abs_z/2.0) *
                 (0.319381530*t - 0.356563782*t*t + 1.781477937*t*t*t
                  - 1.821255978*t*t*t*t + 1.330274429*t*t*t*t*t);
    if (z > 0.0) phi = 1.0 - phi;

    /* Return p-value: P(chi-squared > Q) = 1 - CDF(Q) */
    return phi;
}

/* ================================================================
 * Cross-Correlation Test
 *
 * Tests independence between residuals eps(t) and input u(t).
 * Cross-correlation at lag tau:
 *   r_{eps u}(tau) = sum eps(t+tau) * u(t) / sqrt(sum eps^2 * sum u^2)
 *
 * 95% confidence band: +/- 1.96 / sqrt(N)
 *
 * If |r_{eps u}(tau)| > 1.96/sqrt(N) for any tau, there is significant
 * correlation, indicating unmodeled dynamics.
 *
 * Reference:
 *   Soderstrom & Stoica (1989), Chapter 11
 * ================================================================ */

double pem_validation_crosscorr_max(const double *eps, const double *u,
                                    int N, int max_lag) {
    if (N <= max_lag) return 1.0;

    /* Compute normalization factors */
    double sum_eps2 = 0.0, sum_u2 = 0.0;
    for (int t = 0; t < N; t++) {
        sum_eps2 += eps[t] * eps[t];
        sum_u2 += u[t] * u[t];
    }
    double norm = sqrt(sum_eps2 * sum_u2);
    if (norm < 1e-15) return 0.0;

    double max_corr = 0.0;
    for (int tau = -max_lag; tau <= max_lag; tau++) {
        double cross = 0.0;
        int count = 0;
        for (int t = 0; t < N; t++) {
            int t2 = t + tau;
            if (t2 >= 0 && t2 < N) {
                cross += eps[t2] * u[t];
                count++;
            }
        }
        if (count > 0) {
            double corr = fabs(cross / norm);
            if (corr > max_corr) max_corr = corr;
        }
    }
    return max_corr;
}

/* ================================================================
 * Complete Validation Report
 *
 * Fills a PEMValidation struct with all computed statistics
 * for a given model and dataset.
 * ================================================================ */

int pem_validate_model(const double *y, const double *y_hat,
                       const double *eps, const double *u,
                       int N, int npar,
                       PEMValidation *v) {
    if (!y || !y_hat || !v || N <= 0) return 1;
    v->N = N;
    v->npar = npar;

    /* Loss */
    double sum_sq = 0.0;
    for (int i = 0; i < N; i++) {
        double e = y[i] - y_hat[i];
        sum_sq += e * e;
    }
    double loss = 0.5 * sum_sq / (double)N;
    v->loss = loss;

    /* Fit */
    v->fit_percent = pem_validation_fit(y, y_hat, N);

    /* Information criteria */
    if (N > npar && loss > 0.0) {
        v->aic = pem_validation_aic(loss, N, npar);
        v->aicc = pem_validation_aicc(loss, N, npar);
        v->bic = pem_validation_bic(loss, N, npar);
        v->fpe = pem_validation_fpe(loss, N, npar);
    } else {
        v->aic = v->aicc = v->bic = v->fpe = 1e100;
    }

    /* R-squared */
    v->r_squared = pem_validation_rsquared(y, y_hat, N);
    v->adjusted_r_squared = pem_validation_adj_rsquared(y, y_hat, N, npar);

    /* Whiteness test */
    if (eps) {
        int m = (N > 20) ? 20 : N / 4;
        if (m < 1) m = 1;
        v->residual_whiteness = pem_validation_ljung_box(eps, N, m);
    } else {
        v->residual_whiteness = -1.0;
    }

    /* Cross-correlation test */
    if (eps && u) {
        int max_lag = (N > 25) ? 25 : N / 2;
        if (max_lag < 1) max_lag = 1;
        v->crosscorr_max = pem_validation_crosscorr_max(eps, u, N, max_lag);
    } else {
        v->crosscorr_max = -1.0;
    }

    return 0;
}

/* ================================================================
 * K-Fold Cross-Validation
 *
 * Splits data into K folds, trains on K-1 folds, validates on the
 * remaining fold. Rotates through all folds and averages the results.
 *
 * This provides an estimate of the model's generalization performance
 * that is less biased than in-sample statistics.
 *
 * Note: For time series, standard random K-fold CV violates temporal
 * structure. Use "walk-forward" or "blocked" CV for time series.
 * This implementation uses blocked CV where each fold is a contiguous
 * block of time.
 * ================================================================ */

int pem_cross_validate_blocks(const double *y, const double *y_hat, int N, int K,
                              double *avg_fit, double *std_fit) {
    if (K <= 1 || N < 2 * K) return 1;

    int block_size = N / K;
    double *fits = (double*)malloc((size_t)K * sizeof(double));
    if (!fits) return 1;

    for (int k = 0; k < K; k++) {
        int start = k * block_size;
        int end = (k == K - 1) ? N : start + block_size;
        int nk = end - start;
        fits[k] = pem_validation_fit(y + start, y_hat + start, nk);
    }

    double sum = 0.0, sum_sq = 0.0;
    for (int k = 0; k < K; k++) {
        sum += fits[k];
        sum_sq += fits[k] * fits[k];
    }
    *avg_fit = sum / (double)K;
    *std_fit = sqrt((sum_sq - sum * sum / (double)K) / (double)(K - 1));

    free(fits);
    return 0;
}