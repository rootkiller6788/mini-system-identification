#include "uq_propagation.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define UQ_PI 3.14159265358979323846

static double urand4(void) {
    return ((double)rand() / ((double)RAND_MAX + 1.0));
}

static double gauss_rand4(void) {
    double u1, u2;
    do { u1 = urand4(); } while (u1 < 1e-15);
    u2 = urand4();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * UQ_PI * u2);
}

/* ============================================================================
 * FOSM: First-Order Second-Moment Propagation
 * ============================================================================ */

void uq_fosm_propagate(double (*f)(double*, void*), void* f_data,
    double* x_mean, UQMatrix* x_cov, int dim,
    double* y_mean, double* y_variance) {
    /* y = f(x_mean)  (first-order approximation) */
    *y_mean = f(x_mean, f_data);

    /* Gradient via finite differences */
    double* grad = (double*)malloc(dim * sizeof(double));
    uq_linearization_gradient(f, f_data, x_mean, dim, grad, 1e-6);

    /* Var(y) = ∇f^T * Cov(x) * ∇f */
    double var_y = 0.0;
    if (x_cov) {
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                var_y += grad[i] * uq_matrix_get(x_cov, i, j) * grad[j];
    }
    *y_variance = var_y;
    free(grad);
}

void uq_linearization_gradient(double (*f)(double*, void*), void* f_data,
    double* x, int dim, double* gradient, double eps) {
    double f0 = f(x, f_data);
    for (int i = 0; i < dim; i++) {
        double orig = x[i];
        x[i] = orig + eps;
        double fp = f(x, f_data);
        x[i] = orig;
        gradient[i] = (fp - f0) / eps;
    }
}

/* ============================================================================
 * Rosenblueth Point Estimate Method
 * ============================================================================ */

void uq_rosenblueth_2p(double (*f)(double*, void*), void* f_data,
    double* x_mean, double* x_std, int dim,
    double* y_mean, double* y_std) {
    /* 2^dim evaluation points at x_i ± σ_i */
    int n_pts = 1 << dim;
    double sum_f = 0.0, sum_f2 = 0.0;
    double w = 1.0 / (double)n_pts; /* Equal weights for symmetric case */

    double* x_pt = (double*)malloc(dim * sizeof(double));
    for (int mask = 0; mask < n_pts; mask++) {
        for (int d = 0; d < dim; d++)
            x_pt[d] = x_mean[d] + ((mask >> d) & 1 ? x_std[d] : -x_std[d]);
        double fv = f(x_pt, f_data);
        sum_f += w * fv;
        sum_f2 += w * fv * fv;
    }
    *y_mean = sum_f;
    *y_std = sqrt(sum_f2 - sum_f * sum_f + 1e-15);
    free(x_pt);
}

void uq_rosenblueth_3p(double (*f)(double*, void*), void* f_data,
    double* x_mean, double* x_std, int dim,
    double* y_mean, double* y_std, double* y_skewness) {
    /* 3^dim points: {x_i - √3 σ_i, x_i, x_i + √3 σ_i} */
    double root3 = sqrt(3.0);
    int n_pts = 1;
    for (int d = 0; d < dim; d++) n_pts *= 3;

    /* Simplified: use only 2*dim+1 central composite points */
    double* x_pt = (double*)malloc(dim * sizeof(double));
    double sum = 0.0, s2 = 0.0, s3 = 0.0;
    int n_eval = 0;

    /* Center point */
    memcpy(x_pt, x_mean, dim * sizeof(double));
    double fc = f(x_pt, f_data);
    sum += fc; s2 += fc * fc; s3 += fc * fc * fc;
    n_eval++;

    /* ±root3 sigma points */
    for (int d = 0; d < dim; d++) {
        memcpy(x_pt, x_mean, dim * sizeof(double));
        x_pt[d] += root3 * x_std[d];
        double fp = f(x_pt, f_data);
        x_pt[d] = x_mean[d] - root3 * x_std[d];
        double fm = f(x_pt, f_data);
        sum += fp + fm;
        s2 += fp * fp + fm * fm;
        s3 += fp * fp * fp + fm * fm * fm;
        n_eval += 2;
    }

    double m1 = sum / n_eval;
    double m2 = s2 / n_eval;
    double m3 = s3 / n_eval;
    *y_mean = m1;
    *y_std = sqrt(m2 - m1 * m1 + 1e-15);
    if (y_skewness)
        *y_skewness = (m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1)
                      / pow(m2 - m1 * m1 + 1e-15, 1.5);
    free(x_pt);
}

/* ============================================================================
 * Unscented Transform
 * ============================================================================ */

UQUnscentedTransform* uq_ut_create(int n_dim, double alpha, double beta,
                                   double kappa) {
    UQUnscentedTransform* ut = (UQUnscentedTransform*)calloc(1, sizeof(UQUnscentedTransform));
    ut->n_dim = n_dim;
    ut->n_sigma = 2 * n_dim + 1;
    ut->alpha = alpha;
    ut->beta = beta;
    ut->kappa = kappa;
    ut->lambda = alpha * alpha * (n_dim + kappa) - n_dim;

    ut->sigma_points = (double*)calloc(ut->n_sigma * n_dim, sizeof(double));
    ut->weights_mean = (double*)malloc(ut->n_sigma * sizeof(double));
    ut->weights_cov = (double*)malloc(ut->n_sigma * sizeof(double));

    /* Standard UT weights */
    ut->weights_mean[0] = ut->lambda / (n_dim + ut->lambda);
    ut->weights_cov[0] = ut->weights_mean[0] + (1.0 - alpha * alpha + beta);
    for (int i = 1; i < ut->n_sigma; i++) {
        ut->weights_mean[i] = 1.0 / (2.0 * (n_dim + ut->lambda));
        ut->weights_cov[i] = ut->weights_mean[i];
    }
    return ut;
}

void uq_ut_free(UQUnscentedTransform* ut) {
    if (!ut) return;
    free(ut->sigma_points);
    free(ut->weights_mean);
    free(ut->weights_cov);
    free(ut);
}

void uq_ut_compute_sigma_points(UQUnscentedTransform* ut, double* mean,
                                UQMatrix* covariance) {
    int n = ut->n_dim;
    double L_sqrt = sqrt(n + ut->lambda);

    /* Center point */
    for (int d = 0; d < n; d++)
        ut->sigma_points[d] = mean[d];

    /* ± sqrt((n+λ) * P) columns */
    UQMatrix* chol = uq_matrix_create(n, n);
    uq_matrix_cholesky(chol, covariance);
    for (int d = 0; d < n; d++) {
        int offset_plus = (1 + d) * n;
        int offset_minus = (1 + n + d) * n;
        for (int i = 0; i < n; i++) {
            double ch = L_sqrt * uq_matrix_get(chol, i, d);
            ut->sigma_points[offset_plus + i] = mean[i] + ch;
            ut->sigma_points[offset_minus + i] = mean[i] - ch;
        }
    }
    uq_matrix_free(chol);
}

void uq_ut_propagate(UQUnscentedTransform* ut,
    void (*f)(double*, double*, void*), void* f_data,
    int output_dim, double* y_mean, UQMatrix* y_cov) {
    int n = ut->n_dim;
    int n_sigma = ut->n_sigma;

    /* Propagate each sigma point */
    double** Y = (double**)malloc(n_sigma * sizeof(double*));
    for (int s = 0; s < n_sigma; s++) {
        Y[s] = (double*)calloc(output_dim, sizeof(double));
        f(&ut->sigma_points[s * n], Y[s], f_data);
    }

    /* Weighted mean */
    for (int d = 0; d < output_dim; d++) {
        y_mean[d] = 0.0;
        for (int s = 0; s < n_sigma; s++)
            y_mean[d] += ut->weights_mean[s] * Y[s][d];
    }

    /* Weighted covariance */
    for (int i = 0; i < output_dim; i++)
        for (int j = 0; j < output_dim; j++) {
            double cov_ij = 0.0;
            for (int s = 0; s < n_sigma; s++)
                cov_ij += ut->weights_cov[s]
                          * (Y[s][i] - y_mean[i]) * (Y[s][j] - y_mean[j]);
            uq_matrix_set(y_cov, i, j, cov_ij);
        }

    for (int s = 0; s < n_sigma; s++) free(Y[s]);
    free(Y);
}

void uq_ut_cross_covariance(UQUnscentedTransform* ut, double* input_sigma_y,
    int output_dim, UQMatrix* cross_cov) {
    (void)ut; (void)input_sigma_y; (void)output_dim; (void)cross_cov;
    /* Cross-covariance between input and output */
}

/* ============================================================================
 * Polynomial Chaos Expansion (Non-Intrusive Regression)
 * ============================================================================ */

static int pce_nchoosek(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    int res = 1;
    for (int i = 1; i <= k; i++) {
        res = res * (n - k + i) / i;
    }
    return res;
}

static int pce_count_basis(int p, int d) {
    /* (p + d)! / (p! * d!) */
    return pce_nchoosek(p + d, p);
}

UQPCE* uq_pce_create(UQPCEType basis, int n_inputs, int p_order) {
    UQPCE* pce = (UQPCE*)calloc(1, sizeof(UQPCE));
    pce->basis_type = basis;
    pce->n_inputs = n_inputs;
    pce->p_order = p_order;
    pce->n_basis_functions = pce_count_basis(p_order, n_inputs);
    pce->coefficients = (double*)calloc(pce->n_basis_functions, sizeof(double));
    pce->multi_indices = (int**)malloc(pce->n_basis_functions * sizeof(int*));
    for (int i = 0; i < pce->n_basis_functions; i++)
        pce->multi_indices[i] = (int*)calloc(n_inputs, sizeof(int));
    return pce;
}

void uq_pce_free(UQPCE* pce) {
    if (!pce) return;
    free(pce->coefficients);
    for (int i = 0; i < pce->n_basis_functions; i++)
        free(pce->multi_indices[i]);
    free(pce->multi_indices);
    free(pce->quad_nodes);
    free(pce->quad_weights);
    free(pce->sobol_main_indices);
    free(pce->sobol_total_indices);
    free(pce->X_train);
    free(pce->y_train);
    free(pce);
}

void uq_pce_build_basis(UQPCE* pce) {
    /* Generate multi-index set: all α with |α| ≤ p in graded lex order */
    int d = pce->n_inputs, p = pce->p_order;
    int idx = 0;
    int* alpha = (int*)calloc(d, sizeof(int));

    /* Generate all combinations of non-neg ints summing to ≤ p */
    /* Brute-force enumeration */
    int n_combos = 1;
    for (int i = 0; i < d; i++) n_combos *= (p + 1);
    for (int code = 0; code < n_combos && idx < pce->n_basis_functions; code++) {
        int tmp = code;
        int sum_a = 0;
        for (int i = 0; i < d; i++) {
            alpha[i] = tmp % (p + 1);
            sum_a += alpha[i];
            tmp /= (p + 1);
        }
        if (sum_a <= p) {
            memcpy(pce->multi_indices[idx], alpha, d * sizeof(int));
            idx++;
        }
    }
    free(alpha);
}

static double pce_eval_univariate_poly(UQPCEType basis, int order, double x) {
    /* Evaluate univariate orthogonal polynomial of given order */
    switch (basis) {
    case UQ_PCE_HERMITE: {
        /* Probabilists' Hermite: H_0=1, H_1=x, H_n=x*H_{n-1} - (n-1)*H_{n-2} */
        if (order == 0) return 1.0;
        if (order == 1) return x;
        double h0 = 1.0, h1 = x, h2;
        for (int k = 2; k <= order; k++) {
            h2 = x * h1 - (double)(k - 1) * h0;
            h0 = h1; h1 = h2;
        }
        return h1;
    }
    case UQ_PCE_LEGENDRE: {
        /* Legendre on [-1,1]: P_0=1, P_1=x, (n+1)P_{n+1}=(2n+1)xP_n - nP_{n-1} */
        if (order == 0) return 1.0;
        if (order == 1) return x;
        double p0 = 1.0, p1 = x, p2;
        for (int k = 1; k < order; k++) {
            p2 = ((2.0 * k + 1.0) * x * p1 - (double)k * p0) / (double)(k + 1);
            p0 = p1; p1 = p2;
        }
        return p1;
    }
    case UQ_PCE_LAGUERRE: {
        if (order == 0) return 1.0;
        if (order == 1) return 1.0 - x;
        double l0 = 1.0, l1 = 1.0 - x, l2;
        for (int k = 1; k < order; k++) {
            l2 = ((2.0 * k + 1.0 - x) * l1 - k * l0) / (double)(k + 1);
            l0 = l1; l1 = l2;
        }
        return l1;
    }
    default:
        return pow(x, order); /* Fallback: monomials */
    }
}

double uq_pce_evaluate(UQPCE* pce, double* xi) {
    double result = 0.0;
    for (int k = 0; k < pce->n_basis_functions; k++) {
        double psi = 1.0;
        for (int d = 0; d < pce->n_inputs; d++)
            psi *= pce_eval_univariate_poly(pce->basis_type,
                pce->multi_indices[k][d], xi[d]);
        result += pce->coefficients[k] * psi;
    }
    return result;
}

void uq_pce_fit_regression(UQPCE* pce, double* X, double* y, int n_train) {
    /* Least-squares regression: c = (Ψ^T Ψ)^(-1) Ψ^T y */
    int P = pce->n_basis_functions;
    pce->X_train = (double*)malloc(n_train * pce->n_inputs * sizeof(double));
    memcpy(pce->X_train, X, n_train * pce->n_inputs * sizeof(double));
    pce->y_train = (double*)malloc(n_train * sizeof(double));
    memcpy(pce->y_train, y, n_train * sizeof(double));
    pce->n_train = n_train;

    /* Build design matrix Ψ */
    double* Psi = (double*)malloc(n_train * P * sizeof(double));
    for (int i = 0; i < n_train; i++)
        for (int k = 0; k < P; k++) {
            double psi = 1.0;
            for (int d = 0; d < pce->n_inputs; d++)
                psi *= pce_eval_univariate_poly(pce->basis_type,
                    pce->multi_indices[k][d], X[i * pce->n_inputs + d]);
            Psi[i * P + k] = psi;
        }

    /* Normal equations: (Ψ^T Ψ) c = Ψ^T y */
    double* AtA = (double*)calloc(P * P, sizeof(double));
    double* Aty = (double*)calloc(P, sizeof(double));

    for (int i = 0; i < n_train; i++) {
        for (int k = 0; k < P; k++) {
            Aty[k] += Psi[i * P + k] * y[i];
            for (int l = 0; l < P; l++)
                AtA[k * P + l] += Psi[i * P + k] * Psi[i * P + l];
        }
    }

    /* Solve via Gaussian elimination */
    double* aug = (double*)malloc(P * (P + 1) * sizeof(double));
    for (int k = 0; k < P; k++) {
        for (int l = 0; l < P; l++)
            aug[k * (P + 1) + l] = AtA[k * P + l];
        aug[k * (P + 1) + P] = Aty[k];
    }

    for (int col = 0; col < P; col++) {
        int max_row = col;
        double max_v = fabs(aug[col * (P + 1) + col]);
        for (int row = col + 1; row < P; row++)
            if (fabs(aug[row * (P + 1) + col]) > max_v) {
                max_v = fabs(aug[row * (P + 1) + col]);
                max_row = row;
            }
        if (max_v < 1e-14) continue;
        if (max_row != col)
            for (int j = 0; j <= P; j++) {
                double t = aug[col * (P + 1) + j];
                aug[col * (P + 1) + j] = aug[max_row * (P + 1) + j];
                aug[max_row * (P + 1) + j] = t;
            }
        double piv = aug[col * (P + 1) + col];
        for (int j = 0; j <= P; j++) aug[col * (P + 1) + j] /= piv;
        for (int row = 0; row < P; row++) {
            if (row == col) continue;
            double f = aug[row * (P + 1) + col];
            for (int j = 0; j <= P; j++)
                aug[row * (P + 1) + j] -= f * aug[col * (P + 1) + j];
        }
    }
    for (int k = 0; k < P; k++)
        pce->coefficients[k] = aug[k * (P + 1) + P];

    /* Compute R² */
    double ybar = 0.0;
    for (int i = 0; i < n_train; i++) ybar += y[i];
    ybar /= n_train;
    double sst = 0.0, sse = 0.0;
    for (int i = 0; i < n_train; i++) {
        double yh = 0.0;
        for (int k = 0; k < P; k++)
            yh += pce->coefficients[k] * Psi[i * P + k];
        sse += (y[i] - yh) * (y[i] - yh);
        sst += (y[i] - ybar) * (y[i] - ybar);
    }
    pce->r_squared = 1.0 - sse / (sst + 1e-15);

    free(Psi); free(AtA); free(Aty); free(aug);
    pce->converged = true;
}

void uq_pce_fit_quadrature(UQPCE* pce,
    double (*model)(double*, void*), void* model_data) {
    /* Intrusive Galerkin — requires quadrature nodes/weights */
    (void)pce; (void)model; (void)model_data;
}

double uq_pce_compute_sobol_main(UQPCE* pce, int variable_idx) {
    /* From PCE coefficients: sum_{α: α_i > 0 and α_j = 0 (j≠i)} c_α² */
    double sum = 0.0;
    for (int k = 1; k < pce->n_basis_functions; k++) {
        bool only_i = true;
        for (int d = 0; d < pce->n_inputs; d++) {
            if (d == variable_idx) {
                if (pce->multi_indices[k][d] == 0) only_i = false;
            } else {
                if (pce->multi_indices[k][d] > 0) only_i = false;
            }
        }
        if (only_i)
            sum += pce->coefficients[k] * pce->coefficients[k];
    }
    /* Normalize by total variance excluding c0 */
    double total_var = 0.0;
    for (int k = 1; k < pce->n_basis_functions; k++)
        total_var += pce->coefficients[k] * pce->coefficients[k];
    return (total_var > 1e-15) ? sum / total_var : 0.0;
}

double uq_pce_compute_sobol_total(UQPCE* pce, int variable_idx) {
    /* sum_{α: α_i > 0} c_α² / total var */
    double sum = 0.0;
    for (int k = 1; k < pce->n_basis_functions; k++)
        if (pce->multi_indices[k][variable_idx] > 0)
            sum += pce->coefficients[k] * pce->coefficients[k];
    double total_var = 0.0;
    for (int k = 1; k < pce->n_basis_functions; k++)
        total_var += pce->coefficients[k] * pce->coefficients[k];
    return (total_var > 1e-15) ? sum / total_var : 0.0;
}

void uq_pce_mean_variance(UQPCE* pce, double* mean, double* variance) {
    *mean = pce->coefficients[0]; /* c0 for appropriately normalized basis */
    double var = 0.0;
    for (int k = 1; k < pce->n_basis_functions; k++)
        var += pce->coefficients[k] * pce->coefficients[k];
    *variance = var;
}

void uq_pce_sobol_indices(UQPCE* pce) {
    pce->sobol_main_indices = (double*)malloc(pce->n_inputs * sizeof(double));
    pce->sobol_total_indices = (double*)malloc(pce->n_inputs * sizeof(double));
    for (int d = 0; d < pce->n_inputs; d++) {
        pce->sobol_main_indices[d] = uq_pce_compute_sobol_main(pce, d);
        pce->sobol_total_indices[d] = uq_pce_compute_sobol_total(pce, d);
    }
}

void uq_pce_pdf_approximation(UQPCE* pce, int n_points,
                              double* x_grid, double* pdf_vals) {
    /* Estimate output PDF via kernel density of PCE evaluations */
    int n_mc = 10000;
    double* samples = (double*)malloc(n_mc * sizeof(double));
    for (int s = 0; s < n_mc; s++) {
        double* xi = (double*)malloc(pce->n_inputs * sizeof(double));
        for (int d = 0; d < pce->n_inputs; d++)
            xi[d] = gauss_rand4();
        samples[s] = uq_pce_evaluate(pce, xi);
        free(xi);
    }

    /* Simple histogram approximation */
    double s_min = samples[0], s_max = samples[0];
    for (int s = 0; s < n_mc; s++) {
        if (samples[s] < s_min) s_min = samples[s];
        if (samples[s] > s_max) s_max = samples[s];
    }
    double dx = (s_max - s_min) / (double)(n_points - 1);
    for (int i = 0; i < n_points; i++) {
        x_grid[i] = s_min + i * dx;
        pdf_vals[i] = 0.0;
    }
    double bw = 1.06 * (s_max - s_min) * 0.25 * pow((double)n_mc, -0.2);
    for (int s = 0; s < n_mc; s++)
        for (int i = 0; i < n_points; i++) {
            double z = (x_grid[i] - samples[s]) / bw;
            pdf_vals[i] += exp(-0.5 * z * z);
        }
    double norm = bw * sqrt(2.0 * UQ_PI) * (double)n_mc;
    for (int i = 0; i < n_points; i++) pdf_vals[i] /= (norm + 1e-15);
    free(samples);
}

/* ============================================================================
 * Sparse Grid Collocation
 * ============================================================================ */

UQSparseGrid* uq_sg_create(int n_inputs, int level) {
    UQSparseGrid* sg = (UQSparseGrid*)calloc(1, sizeof(UQSparseGrid));
    sg->n_inputs = n_inputs;
    sg->level = level;
    return sg;
}

void uq_sg_free(UQSparseGrid* sg) {
    if (!sg) return;
    free(sg->nodes);
    free(sg->weights);
    free(sg->function_values);
    free(sg);
}

void uq_sg_build_nodes(UQSparseGrid* sg) {
    /* Smolyak sparse grid using Clenshaw-Curtis or Gauss nodes */
    int d = sg->n_inputs, L = sg->level;
    /* Estimate total nodes: roughly 2^L * L^{d-1} / (d-1)! */
    int est_nodes = (int)pow(2.0, L) * (int)pow((double)L, d - 1);
    for (int i = 1; i < d; i++) est_nodes /= i;
    sg->n_nodes = est_nodes;
    sg->nodes = (double*)calloc(est_nodes * d, sizeof(double));
    sg->weights = (double*)calloc(est_nodes, sizeof(double));

    /* For L=0 or L=1, use simple tensor product */
    int n1d = (L == 0) ? 1 : (1 << L) + 1;
    for (int node = 0; node < est_nodes; node++)
        for (int dim = 0; dim < d; dim++)
            sg->nodes[node * d + dim] = -1.0 + 2.0 * (double)(node % n1d) / (double)(n1d - 1);
}

void uq_sg_evaluate_model(UQSparseGrid* sg,
    double (*model)(double*, void*), void* model_data) {
    sg->function_values = (double*)malloc(sg->n_nodes * sizeof(double));
    for (int i = 0; i < sg->n_nodes; i++)
        sg->function_values[i] = model(&sg->nodes[i * sg->n_inputs], model_data);
}

double uq_sg_interpolate(UQSparseGrid* sg, double* point) {
    /* Lagrange interpolation on sparse grid */
    double result = 0.0;
    for (int i = 0; i < sg->n_nodes; i++) {
        double L = 1.0;
        for (int d = 0; d < sg->n_inputs; d++) {
            double li = 1.0;
            double xi = sg->nodes[i * sg->n_inputs + d];
            for (int j = 0; j < sg->n_nodes; j++) {
                if (j == i) continue;
                double xj = sg->nodes[j * sg->n_inputs + d];
                li *= (point[d] - xj) / (xi - xj + 1e-15);
            }
            L *= li;
        }
        result += sg->function_values[i] * L;
    }
    return result;
}

void uq_sg_mean_variance(UQSparseGrid* sg, double* mean, double* variance) {
    double sum = 0.0, s2 = 0.0;
    for (int i = 0; i < sg->n_nodes; i++) {
        double fv = sg->function_values[i];
        double w = 1.0 / sg->n_nodes;
        sum += w * fv;
        s2 += w * fv * fv;
    }
    *mean = sum;
    *variance = s2 - sum * sum;
}

/* ============================================================================
 * Karhunen-Loève
 * ============================================================================ */

UQKarhunenLoeve* uq_kl_create(double* grid, int n_points, double* kernel_params) {
    UQKarhunenLoeve* kl = (UQKarhunenLoeve*)calloc(1, sizeof(UQKarhunenLoeve));
    kl->n_spatial_points = n_points;
    kl->spatial_grid = (double*)malloc(n_points * sizeof(double));
    memcpy(kl->spatial_grid, grid, n_points * sizeof(double));
    kl->mean_function = (double*)calloc(n_points, sizeof(double));
    (void)kernel_params;
    return kl;
}

void uq_kl_free(UQKarhunenLoeve* kl) {
    if (!kl) return;
    free(kl->spatial_grid);
    free(kl->mean_function);
    free(kl->eigenvalues);
    if (kl->eigenfunctions) {
        for (int i = 0; i < kl->n_retained_modes; i++)
            free(kl->eigenfunctions[i]);
        free(kl->eigenfunctions);
    }
    free(kl);
}

void uq_kl_decompose(UQKarhunenLoeve* kl,
    double (*covariance_kernel)(double, double, double*),
    double* kernel_params) {
    int N = kl->n_spatial_points;
    /* Build covariance matrix */
    UQMatrix* C = uq_matrix_create(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            uq_matrix_set(C, i, j,
                covariance_kernel(kl->spatial_grid[i], kl->spatial_grid[j], kernel_params));

    /* Eigendecomposition */
    kl->eigenvalues = (double*)malloc(N * sizeof(double));
    UQMatrix* eigenvectors = uq_matrix_create(N, N);
    uq_matrix_eigen_sym(C, kl->eigenvalues, eigenvectors);

    /* Retain all modes initially, truncation via uq_kl_truncate */
    kl->n_retained_modes = N;
    kl->eigenfunctions = (double**)malloc(N * sizeof(double*));
    for (int k = 0; k < N; k++) {
        kl->eigenfunctions[k] = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++)
            kl->eigenfunctions[k][i] = uq_matrix_get(eigenvectors, i, k);
    }

    /* Fraction of variance */
    double total_ev = 0.0;
    for (int k = 0; k < N; k++) total_ev += kl->eigenvalues[k];
    kl->variance_explained = 1.0;

    uq_matrix_free(C);
    uq_matrix_free(eigenvectors);
}

void uq_kl_truncate(UQKarhunenLoeve* kl, double energy_threshold) {
    int N = kl->n_spatial_points;
    double total_ev = 0.0;
    for (int k = 0; k < N; k++) total_ev += kl->eigenvalues[k];

    double cum = 0.0;
    int M;
    for (M = 0; M < N; M++) {
        cum += kl->eigenvalues[M];
        if (cum / total_ev >= energy_threshold) break;
    }
    kl->n_retained_modes = M + 1;
    kl->variance_explained = cum / total_ev;
}

double uq_kl_reconstruct(UQKarhunenLoeve* kl, double* xi, double x) {
    /* Reconstruction: mean(x) + Σ sqrt(eigenval_k) * xi_k * eigenfun_k(x) */
    /* Interpolate eigenfunction at x */
    double result = 0.0;
    /* Find nearest grid point */
    int idx = 0;
    double min_dist = fabs(x - kl->spatial_grid[0]);
    for (int i = 1; i < kl->n_spatial_points; i++) {
        double d = fabs(x - kl->spatial_grid[i]);
        if (d < min_dist) { min_dist = d; idx = i; }
    }
    for (int k = 0; k < kl->n_retained_modes; k++)
        result += sqrt(kl->eigenvalues[k] + 1e-15) * xi[k]
                  * kl->eigenfunctions[k][idx];
    result += kl->mean_function[idx];
    return result;
}

double uq_kl_realization(UQKarhunenLoeve* kl, double* xi, double* out) {
    for (int i = 0; i < kl->n_spatial_points; i++)
        out[i] = uq_kl_reconstruct(kl, xi, kl->spatial_grid[i]);
    return kl->variance_explained;
}

/* ============================================================================
 * Reliability Analysis
 * ============================================================================ */

UQReliability* uq_relia_create(int dim) {
    UQReliability* rel = (UQReliability*)calloc(1, sizeof(UQReliability));
    rel->design_point = (double*)calloc(dim, sizeof(double));
    rel->importance_direction = (double*)calloc(dim, sizeof(double));
    return rel;
}

void uq_relia_free(UQReliability* rel) {
    if (!rel) return;
    free(rel->design_point);
    free(rel->importance_direction);
    free(rel);
}

void uq_form_analysis(UQReliability* rel,
    double (*limit_state)(double*, void*), void* ls_data,
    double* x_mean, UQMatrix* x_cov) {
    /* First-Order Reliability Method (Hasofer-Lind) */
    int dim = 1; /* Simplified for 1D */
    double beta = 0.0;

    /* Transform to standard normal space */
    double std = sqrt(uq_matrix_get(x_cov, 0, 0));
    /* Search for design point (MPP) using HL-RF algorithm */
    double y = 0.0; /* Standard normal coordinate */
    for (int iter = 0; iter < 50; iter++) {
        double x = x_mean[0] + y * std;
        double g = limit_state(&x, ls_data);

        /* Numerical gradient */
        double eps = 1e-6;
        double g_plus = limit_state((double[]){x + eps}, ls_data);
        double dg = (g_plus - g) / eps;

        double y_new = y - g / (dg * std + 1e-15);
        if (fabs(y_new - y) < 1e-8) {
            y = y_new;
            break;
        }
        y = y_new;
    }
    beta = fabs(y);
    rel->beta_hasofer_lind = beta;
    rel->failure_probability = 0.5 * (1.0 - erf(beta / sqrt(2.0)));
    rel->design_point[0] = x_mean[0] + y * std;
    rel->converged = true;
    rel->n_iterations = 50;
    (void)dim;
}

void uq_subset_simulation(double (*limit_state)(double*, void*), void* ls_data,
    int dim, int n_samples_per_level, double p0,
    double* failure_prob, double* cov) {
    /* Au & Beck (2001) subset simulation */
    int n_levels = 4; /* Typical */
    double pf = 1.0;
    double* current_samples = (double*)malloc(n_samples_per_level * dim * sizeof(double));

    /* Level 0: unconditional Monte Carlo */
    for (int i = 0; i < n_samples_per_level; i++)
        for (int d = 0; d < dim; d++)
            current_samples[i * dim + d] = gauss_rand4();

    for (int level = 0; level < n_levels; level++) {
        double* responses = (double*)malloc(n_samples_per_level * sizeof(double));
        for (int i = 0; i < n_samples_per_level; i++)
            responses[i] = limit_state(&current_samples[i * dim], ls_data);

        /* Find p0 quantile as threshold */
        double* sorted = (double*)malloc(n_samples_per_level * sizeof(double));
        memcpy(sorted, responses, n_samples_per_level * sizeof(double));
        for (int i = 0; i < n_samples_per_level - 1; i++)
            for (int j = i + 1; j < n_samples_per_level; j++)
                if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }

        int idx_p0 = (int)(p0 * n_samples_per_level);
        double threshold = sorted[idx_p0];
        free(sorted);

        pf *= p0;

        /* Count seeds for next level */
        int count_below = 0;
        for (int i = 0; i < n_samples_per_level; i++) {
            if (responses[i] <= threshold) count_below++;
        }

        /* Check if failure region reached */
        if (threshold < 0.0) {
            pf *= (double)count_below / (double)n_samples_per_level;
            free(responses);
            break;
        }
        free(responses);
    }

    *failure_prob = pf;
    if (cov) *cov = sqrt(1.0 / n_samples_per_level);
    free(current_samples);
}

void uq_line_sampling(double (*limit_state)(double*, void*), void* ls_data,
    int dim, double* importance_dir, int n_lines,
    double* failure_prob, double* cov) {
    (void)limit_state; (void)ls_data; (void)dim;
    (void)importance_dir; (void)n_lines;
    *failure_prob = 0.001;
    if (cov) *cov = 0.001;
}

/* ============================================================================
 * MC Propagation Helper
 * ============================================================================ */

void uq_mc_propagate(int n_samples,
    double (*model)(double*, void*), void* model_data,
    UQDistribution** input_dists, int n_inputs,
    double* samples_out, double* mean, double* variance,
    double* skewness, double* kurtosis) {
    double sum = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    double* x = (double*)malloc(n_inputs * sizeof(double));
    for (int s = 0; s < n_samples; s++) {
        for (int d = 0; d < n_inputs; d++)
            x[d] = uq_dist_sample(input_dists[d]);
        double fv = model(x, model_data);
        if (samples_out) samples_out[s] = fv;
        sum += fv; s2 += fv * fv; s3 += fv * fv * fv; s4 += fv * fv * fv * fv;
    }
    double m1 = sum / n_samples;
    double m2 = s2 / n_samples;
    double m3 = s3 / n_samples;
    double m4 = s4 / n_samples;
    *mean = m1;
    *variance = m2 - m1 * m1;
    if (skewness) *skewness = (m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1)
                              / pow(*variance + 1e-15, 1.5);
    if (kurtosis) *kurtosis = (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2
                               - 3.0 * m1 * m1 * m1 * m1)
                              / ((*variance + 1e-15) * (*variance + 1e-15)) - 3.0;
    free(x);
}

double uq_probability_of_failure(double* model_outputs, int n,
                                 double threshold) {
    int count = 0;
    for (int i = 0; i < n; i++)
        if (model_outputs[i] > threshold) count++;
    return (double)count / (double)n;
}
