#include "nlsid_algorithms.h"
#include "nlsid_models.h"
#include "nlsid_validation.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Part 1: Cost Function and Residuals
 * ============================================================================ */

int nlsid_compute_residuals(const NLSIDModel* model,
                             const NLSIDDataset* ds,
                             double** e, int* n) {
    if (!model || !ds || !e || !n) return -1;
    int n_samples = ds->n_samples;
    int max_lag = 10;
    int n_eff = n_samples - max_lag;
    if (n_eff <= 0) n_eff = n_samples;
    *n = n_eff;
    *e = (double*)calloc((size_t)n_eff, sizeof(double));
    if (!*e) return -1;

    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;

    for (int t = max_lag; t < n_samples; t++) {
        double y_hat = 0.0;
        if (model->predict_one_step) {
            double y_h; model->predict_one_step(
                (NLSIDModel*)model, u, y, t, &y_h); y_hat = y_h;
        }
        (*e)[t - max_lag] = y[t] - y_hat;
    }
    return 0;
}

double nlsid_compute_cost(const NLSIDModel* model,
                           const NLSIDDataset* ds,
                           const NLSIDConfig* config) {
    if (!model || !ds) return 1e100;
    int n_samples = ds->n_samples;
    if (n_samples <= 0) return 1e100;

    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;

    double cost = 0.0;
    int count = 0;
    int start_t = 10;

    for (int t = start_t; t < n_samples; t++) {
        double y_hat = 0.0;
        if (model->predict_one_step) {
            double y_h; model->predict_one_step((NLSIDModel*)model, u, y, t, &y_h); y_hat = y_h;
        }
        double e_t = y[t] - y_hat;
        switch (config->cost_type) {
            case COST_ABSOLUTE: cost += fabs(e_t); break;
            case COST_HUBER: cost += nlsid_huber_loss(e_t, config->huber_delta); break;
            case COST_EPSILON_INSENSITIVE:
                cost += nlsid_epsilon_insensitive_loss(e_t, config->epsilon_ins);
                break;
            default: cost += e_t * e_t; break;
        }
        count++;
    }
    if (count == 0) return 1e100;
    return cost / (double)count;
}

double nlsid_compute_cost_regularized(const NLSIDModel* model,
                                       const NLSIDDataset* ds,
                                       const NLSIDConfig* config,
                                       double lambda) {
    double v_data = nlsid_compute_cost(model, ds, config);
    double penalty = 0.0;
    if (model->params && lambda > 0.0) {
        for (int i = 0; i < model->n_params; i++)
            penalty += model->params[i] * model->params[i];
        penalty *= lambda;
    }
    return v_data + penalty;
}

double nlsid_huber_loss(double e, double delta) {
    double abs_e = fabs(e);
    if (abs_e <= delta) return 0.5 * e * e;
    return delta * (abs_e - 0.5 * delta);
}

double nlsid_epsilon_insensitive_loss(double e, double eps) {
    double abs_e = fabs(e);
    if (abs_e <= eps) return 0.0;
    return abs_e - eps;
}

/* ============================================================================
 * Part 2: Numerical Differentiation (Gradient and Jacobian)  [L4 Theorem]
 * ============================================================================ */

int nlsid_compute_gradient_fd(const NLSIDModel* model,
                               const NLSIDDataset* ds,
                               const NLSIDConfig* config,
                               double* gradient, int n_params) {
    if (!model || !ds || !gradient || n_params <= 0) return -1;

    double* params_save = (double*)malloc((size_t)n_params * sizeof(double));
    if (!params_save) return -1;
    model->get_params((NLSIDModel*)model, params_save);

    /* Central finite differences: dV/dθ_j ≈ (V(θ+h*e_j) - V(θ-h*e_j)) / (2h) */
    double h = 1e-6;
    for (int j = 0; j < n_params; j++) {
        double* params = (double*)malloc((size_t)n_params * sizeof(double));
        memcpy(params, params_save, (size_t)n_params * sizeof(double));

        params[j] += h;
        model->set_params((NLSIDModel*)model, params);
        double v_plus = nlsid_compute_cost(model, ds, config);

        params[j] = params_save[j] - h;
        model->set_params((NLSIDModel*)model, params);
        double v_minus = nlsid_compute_cost(model, ds, config);

        gradient[j] = (v_plus - v_minus) / (2.0 * h);
        free(params);
    }

    model->set_params((NLSIDModel*)model, params_save);
    free(params_save);
    return 0;
}

int nlsid_compute_gradient_narx(const NARXModel* narx,
                                 const NLSIDDataset* ds,
                                 double* gradient, int n_params) {
    if (!narx || !ds || !gradient || n_params <= 0) return -1;
    if (!narx->expansion) return -1;

    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;
    int N = ds->n_samples;
    int ny = narx->ny, nu = narx->nu, nk = narx->nk;

    /* For linear-in-parameters basis expansion:
     * y_hat(t) = Σ w_i φ_i(x(t))
     * dV/dw_j = -(2/N) Σ e(t) φ_j(x(t))  */
    for (int j = 0; j < n_params; j++) gradient[j] = 0.0;

    int start_t = (ny > nu + nk) ? ny : nu + nk;
    int count = 0;
    for (int t = start_t; t < N; t++) {
        /* Build regressor */
        int reg_dim;
        double* phi = nlsid_build_regressor(y, ny, u, nu, nk, t, &reg_dim);
        if (!phi) continue;

        double y_hat = basis_expansion_eval(narx->expansion, phi);
        double e_t = y[t] - y_hat;

        /* Jacobian row: J[j] = φ_j(x) for j=1..n_params-1, J[0]=1 (offset) */
        gradient[0] += -2.0 * e_t; /* dV/d(offset) */
        for (int j = 1; j < n_params && j <= narx->n_params; j++) {
            int basis_idx = j - 1;
            if (basis_idx < narx->expansion->n_bases) {
                gradient[j] += -2.0 * e_t *
                    basis_evaluate(narx->expansion->bases[basis_idx], phi);
            }
        }
        free(phi);
        count++;
    }
    if (count > 0)
        for (int j = 0; j < n_params; j++) gradient[j] /= (double)count;

    return 0;
}

int nlsid_compute_hessian_gn(const NLSIDModel* model,
                              const NLSIDDataset* ds,
                              const NLSIDConfig* config,
                              double** H, int n_params) {
    if (!model || !ds || !H || n_params <= 0) return -1;
    if (!model->predict_one_step) return -1;

    *H = (double*)calloc((size_t)(n_params * n_params), sizeof(double));
    if (!*H) return -1;

    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;
    int N = ds->n_samples;

    /* Use finite differences to compute Jacobian rows */
    double* params_save = (double*)malloc((size_t)n_params * sizeof(double));
    model->get_params((NLSIDModel*)model, params_save);

    double h = 1e-6;
    for (int j = 0; j < n_params; j++) {
        double* params_p = (double*)malloc((size_t)n_params * sizeof(double));
        double* params_m = (double*)malloc((size_t)n_params * sizeof(double));
        memcpy(params_p, params_save, (size_t)n_params * sizeof(double));
        memcpy(params_m, params_save, (size_t)n_params * sizeof(double));
        params_p[j] += h; params_m[j] -= h;

        for (int t = 20; t < N; t++) {
            double yp, ym;
            model->set_params((NLSIDModel*)model, params_p);
            model->predict_one_step((NLSIDModel*)model, u, y, t, &yp);
            double e_plus = y[t] - yp;
            model->set_params((NLSIDModel*)model, params_m);
            model->predict_one_step((NLSIDModel*)model, u, y, t, &ym);
            double e_minus = y[t] - ym;

            double J_tj = (e_plus - e_minus) / (2.0 * h);

            for (int i = 0; i < n_params; i++) {
                model->set_params((NLSIDModel*)model, params_save);
                /* Recompute J_ti */
                double* pp = (double*)malloc((size_t)n_params * sizeof(double));
                double* mm = (double*)malloc((size_t)n_params * sizeof(double));
                memcpy(pp, params_save, (size_t)n_params * sizeof(double));
                memcpy(mm, params_save, (size_t)n_params * sizeof(double));
                pp[i] += h; mm[i] -= h;

                double ep, em;
                model->set_params((NLSIDModel*)model, pp);
                model->predict_one_step((NLSIDModel*)model, u, y, t, &ep);
                double ep_err = y[t] - ep;
                model->set_params((NLSIDModel*)model, mm);
                model->predict_one_step((NLSIDModel*)model, u, y, t, &em);
                double em_err = y[t] - em;
                double J_ti = (ep_err - em_err) / (2.0 * h);

                (*H)[i * n_params + j] += J_ti * J_tj;
                free(pp); free(mm);
            }
        }
        free(params_p); free(params_m);
    }

    model->set_params((NLSIDModel*)model, params_save);
    free(params_save);
    return 0;
}

int nlsid_compute_jacobian(const NLSIDModel* model,
                            const NLSIDDataset* ds,
                            double*** J, int* n_rows, int* n_cols) {
    if (!model || !ds || !J || !n_rows || !n_cols) return -1;

    int n_params = model->n_params;
    *n_cols = n_params;
    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;
    int N = ds->n_samples;
    int start_t = 10;
    *n_rows = N - start_t;
    if (*n_rows <= 0) return -1;

    double** jac = (double**)malloc((size_t)(*n_rows) * sizeof(double*));
    if (!jac) return -1;

    double* params_save = (double*)malloc((size_t)n_params * sizeof(double));
    model->get_params((NLSIDModel*)model, params_save);

    double h = 1e-6;
    for (int t = start_t; t < N; t++) {
        int row = t - start_t;
        jac[row] = (double*)calloc((size_t)n_params, sizeof(double));

        for (int j = 0; j < n_params; j++) {
            double* params = (double*)malloc((size_t)n_params * sizeof(double));
            memcpy(params, params_save, (size_t)n_params * sizeof(double));
            params[j] += h;
            model->set_params((NLSIDModel*)model, params);
            double ep; model->predict_one_step((NLSIDModel*)model, u, y, t, &ep);
            params[j] = params_save[j] - h;
            model->set_params((NLSIDModel*)model, params);
            double em; model->predict_one_step((NLSIDModel*)model, u, y, t, &em);
            jac[row][j] = (ep - em) / (2.0 * h);
            free(params);
        }
    }

    model->set_params((NLSIDModel*)model, params_save);
    free(params_save);
    *J = jac;
    return 0;
}
/* ============================================================================
 * Part 3: Optimization Algorithms
 * ============================================================================ */

int nlsid_optimize_gauss_newton(NLSIDModel* model,
                                 const NLSIDDataset* ds,
                                 NLSIDConfig* config,
                                 NLSIDResult* result) {
    if (!model || !ds || !config || !result) return -1;
    int n_params = model->n_params;
    if (n_params <= 0) return -1;

    double* params = (double*)malloc((size_t)n_params * sizeof(double));
    model->get_params(model, params);

    double lambda = config->lambda;
    double prev_cost = nlsid_compute_cost(model, ds, config);

    for (int iter = 0; iter < config->max_iterations; iter++) {
        /* Compute Jacobian J and residuals e */
        double** J = NULL; int n_rows, n_cols;
        nlsid_compute_jacobian(model, ds, &J, &n_rows, &n_cols);
        if (!J || n_rows <= 0) { free(params); return -1; }

        double* e = (double*)malloc((size_t)n_rows * sizeof(double));
        const double* u = ds->input->channels[0]->data;
        const double* y = ds->output->channels[0]->data;
        for (int t = 0; t < n_rows; t++) {
            double yh; model->predict_one_step(model, u, y, t + 10, &yh);
            e[t] = y[t + 10] - yh;
        }

        /* Compute J^T J and J^T e */
        double* JTJ = (double*)calloc((size_t)(n_params * n_params), sizeof(double));
        double* JTe = (double*)calloc((size_t)n_params, sizeof(double));

        for (int i = 0; i < n_params; i++) {
            for (int t = 0; t < n_rows; t++) JTe[i] += J[t][i] * e[t];
            for (int j = 0; j < n_params; j++)
                for (int t = 0; t < n_rows; t++)
                    JTJ[i * n_params + j] += J[t][i] * J[t][j];
        }

        /* Add damping */
        for (int i = 0; i < n_params; i++)
            JTJ[i * n_params + i] += lambda;

        /* Solve JTJ * delta = -JTe */
        double* delta = (double*)malloc((size_t)n_params * sizeof(double));
        int solve_ok = nlsid_solve_linear_system(JTJ, JTe, n_params, delta);
        if (solve_ok != 0) {
            /* Fallback: steepest descent step */
            for (int i = 0; i < n_params; i++) delta[i] = -JTe[i] * 0.01;
        }

        /* Update: θ = θ + delta */
        for (int i = 0; i < n_params; i++) params[i] += delta[i];
        model->set_params(model, params);

        double new_cost = nlsid_compute_cost(model, ds, config);

        if (new_cost < prev_cost) {
            lambda /= config->lambda_decay;
            if (fabs(prev_cost - new_cost) < config->tolerance) {
                prev_cost = new_cost;
                free(e); free(JTJ); free(JTe); free(delta);
                for (int t = 0; t < n_rows; t++) free(J[t]);
                free(J);
                result->iterations_used = iter + 1;
                result->converged = true;
                break;
            }
        } else {
            lambda *= config->lambda_decay;
            for (int i = 0; i < n_params; i++) params[i] -= delta[i];
            model->set_params(model, params);
        }
        prev_cost = new_cost;

        free(e); free(JTJ); free(JTe); free(delta);
        for (int t = 0; t < n_rows; t++) { free(J[t]); }
        free(J);

        if (config->verbose && (iter % config->print_interval == 0))
            printf("  GN iter %d: cost=%.6e, lambda=%.2e\n", iter, prev_cost, lambda);

        result->iterations_used = iter + 1;
    }

    result->final_cost = prev_cost;
    free(params);
    return 0;
}

int nlsid_optimize_levenberg_marquardt(NLSIDModel* model,
                                        const NLSIDDataset* ds,
                                        NLSIDConfig* config,
                                        NLSIDResult* result) {
    if (!model || !ds || !config || !result) return -1;
    int n_params = model->n_params;
    if (n_params <= 0) return -1;

    double* params = (double*)malloc((size_t)n_params * sizeof(double));
    model->get_params(model, params);

    double lambda = config->lambda;
    double cost = nlsid_compute_cost(model, ds, config);

    for (int iter = 0; iter < config->max_iterations; iter++) {
        double** J = NULL; int n_rows, n_cols;
        nlsid_compute_jacobian(model, ds, &J, &n_rows, &n_cols);
        if (!J || n_rows <= 0) { free(params); return -1; }

        double* e = (double*)malloc((size_t)n_rows * sizeof(double));
        const double* u = ds->input->channels[0]->data;
        const double* y = ds->output->channels[0]->data;
        int start_t = 10;
        for (int t = 0; t < n_rows; t++) {
            double yh; model->predict_one_step(model, u, y, t + start_t, &yh);
            e[t] = y[t + start_t] - yh;
        }

        /* Build normal equations: (J^T J + λ diag(J^T J)) δ = J^T e */
        double* JTJ = (double*)calloc((size_t)(n_params * n_params), sizeof(double));
        double* JTe = (double*)calloc((size_t)n_params, sizeof(double));

        for (int i = 0; i < n_params; i++) {
            for (int t = 0; t < n_rows; t++) JTe[i] += J[t][i] * e[t];
            for (int j = 0; j < n_params; j++)
                for (int t = 0; t < n_rows; t++)
                    JTJ[i * n_params + j] += J[t][i] * J[t][j];
        }

        /* LM damping: augment diagonal with λ * diag(J^T J) */
        for (int i = 0; i < n_params; i++)
            JTJ[i * n_params + i] *= (1.0 + lambda);

        double* delta = (double*)malloc((size_t)n_params * sizeof(double));
        if (nlsid_solve_linear_system(JTJ, JTe, n_params, delta) != 0) {
            /* Fallback: gradient descent */
            for (int i = 0; i < n_params; i++) delta[i] = -JTe[i] * config->step_size_init;
            for (int i = 0; i < n_params; i++) delta[i] /= (1.0 + lambda);
        }

        /* Trial step */
        double* trial_params = (double*)malloc((size_t)n_params * sizeof(double));
        for (int i = 0; i < n_params; i++) trial_params[i] = params[i] + delta[i];
        model->set_params(model, trial_params);
        double new_cost = nlsid_compute_cost(model, ds, config);

        /* Predicted reduction: q(0) - q(δ) = -g^T δ - 0.5 δ^T J^T J δ */
        double pred_reduction = 0.0;
        for (int i = 0; i < n_params; i++) {
            pred_reduction -= JTe[i] * delta[i];
            for (int j = 0; j < n_params; j++)
                pred_reduction -= 0.5 * delta[i] * JTJ[i * n_params + j] * delta[j];
        }
        /* Scale predicted reduction by λ */
        pred_reduction += 0.5 * lambda;
        for (int i = 0; i < n_params; i++)
            pred_reduction -= 0.5 * lambda * delta[i] * delta[i];

        double actual_reduction = cost - new_cost;
        double ratio = (fabs(pred_reduction) > 1e-15)
                        ? actual_reduction / pred_reduction : 0.0;

        if (ratio > 0.0) {
            /* Accept step */
            memcpy(params, trial_params, (size_t)n_params * sizeof(double));
            cost = new_cost;
            lambda *= (1.0 / 3.0 > 1.0 - pow(2.0*ratio - 1.0, 3.0))
                      ? (1.0 / 3.0) : (1.0 - pow(2.0*ratio - 1.0, 3.0));
            if (lambda < 1e-10) lambda = 1e-10;

            if (fabs(actual_reduction) < config->tolerance) {
                free(e); free(JTJ); free(JTe); free(delta); free(trial_params);
                for (int t = 0; t < n_rows; t++) { free(J[t]); } free(J);
                result->iterations_used = iter + 1;
                result->converged = true;
                break;
            }
        } else {
            /* Reject step */
            model->set_params(model, params);
            lambda *= 2.0;
            if (lambda > 1e10) lambda = 1e10;
        }

        free(e); free(JTJ); free(JTe); free(delta); free(trial_params);
        for (int t = 0; t < n_rows; t++) { free(J[t]); } free(J);

        if (config->verbose && (iter % config->print_interval == 0))
            printf("  LM iter %d: cost=%.6e, lambda=%.2e, ratio=%.4f\n",
                   iter, cost, lambda, ratio);

        result->iterations_used = iter + 1;
    }

    result->final_cost = cost;
    result->converged = result->converged;
    free(params);
    return 0;
}

int nlsid_optimize_steepest_descent(NLSIDModel* model,
                                     const NLSIDDataset* ds,
                                     NLSIDConfig* config,
                                     NLSIDResult* result) {
    if (!model || !ds || !config || !result) return -1;
    int n_params = model->n_params;
    if (n_params <= 0) return -1;

    double* params = (double*)malloc((size_t)n_params * sizeof(double));
    model->get_params(model, params);
    double* gradient = (double*)malloc((size_t)n_params * sizeof(double));

    double cost = nlsid_compute_cost(model, ds, config);
    result->iterations_used = 0;

    for (int iter = 0; iter < config->max_iterations; iter++) {
        nlsid_compute_gradient_fd(model, ds, config, gradient, n_params);

        double grad_norm = 0.0;
        for (int i = 0; i < n_params; i++) grad_norm += gradient[i] * gradient[i];
        grad_norm = sqrt(grad_norm);

        if (grad_norm < config->tolerance) {
            result->converged = true;
            result->iterations_used = iter + 1;
            break;
        }

        /* Line search to find step size */
        double* direction = (double*)malloc((size_t)n_params * sizeof(double));
        for (int i = 0; i < n_params; i++) direction[i] = -gradient[i];

        double alpha = nlsid_line_search_armijo(model, ds, config,
            params, direction, gradient, cost);

        for (int i = 0; i < n_params; i++) params[i] += alpha * direction[i];
        model->set_params(model, params);

        double new_cost = nlsid_compute_cost(model, ds, config);
        if (fabs(cost - new_cost) < config->tolerance) {
            result->converged = true;
            result->iterations_used = iter + 1;
            free(direction);
            break;
        }
        cost = new_cost;

        if (config->verbose && (iter % config->print_interval == 0))
            printf("  SD iter %d: cost=%.6e, |grad|=%.2e, alpha=%.4f\n",
                   iter, cost, grad_norm, alpha);

        free(direction);
        result->iterations_used = iter + 1;
    }

    result->final_cost = cost;
    result->final_gradient_norm = 0.0;
    for (int i = 0; i < n_params; i++)
        result->final_gradient_norm += gradient[i] * gradient[i];
    result->final_gradient_norm = sqrt(result->final_gradient_norm);

    free(params); free(gradient);
    return 0;
}
/* ============================================================================
 * Part 3b: Conjugate Gradient (Fletcher-Reeves)
 * ============================================================================ */

int nlsid_optimize_conjugate_gradient(NLSIDModel* model,
                                       const NLSIDDataset* ds,
                                       NLSIDConfig* config,
                                       NLSIDResult* result) {
    if (!model || !ds || !config || !result) return -1;
    int n_params = model->n_params;
    if (n_params <= 0) return -1;

    double* params = (double*)malloc((size_t)n_params * sizeof(double));
    model->get_params(model, params);
    double* grad = (double*)malloc((size_t)n_params * sizeof(double));
    double* grad_old = (double*)malloc((size_t)n_params * sizeof(double));
    double* direction = (double*)malloc((size_t)n_params * sizeof(double));

    nlsid_compute_gradient_fd(model, ds, config, grad, n_params);
    for (int i = 0; i < n_params; i++) direction[i] = -grad[i];

    double cost = nlsid_compute_cost(model, ds, config);
    result->iterations_used = 0;

    for (int iter = 0; iter < config->max_iterations; iter++) {
        /* Save old gradient for beta computation */
        for (int i = 0; i < n_params; i++) grad_old[i] = grad[i];

        /* Line search along direction */
        double alpha = nlsid_line_search_armijo(model, ds, config,
            params, direction, grad, cost);

        /* Update */
        for (int i = 0; i < n_params; i++) params[i] += alpha * direction[i];
        model->set_params(model, params);

        double new_cost = nlsid_compute_cost(model, ds, config);
        if (fabs(cost - new_cost) < config->tolerance) {
            result->converged = true;
            result->iterations_used = iter + 1;
            break;
        }
        cost = new_cost;

        /* Compute new gradient */
        nlsid_compute_gradient_fd(model, ds, config, grad, n_params);

        /* Fletcher-Reeves beta */
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n_params; i++) {
            num += grad[i] * grad[i];
            den += grad_old[i] * grad_old[i];
        }
        double beta = (den > 1e-15) ? num / den : 0.0;
        /* Reset if negative or too large */
        if (beta < 0.0 || beta > 10.0) beta = 0.0;

        for (int i = 0; i < n_params; i++)
            direction[i] = -grad[i] + beta * direction[i];

        if (config->verbose && (iter % config->print_interval == 0))
            printf("  CG iter %d: cost=%.6e\n", iter, cost);

        result->iterations_used = iter + 1;
    }

    result->final_cost = cost;
    result->final_gradient_norm = 0.0;
    for (int i = 0; i < n_params; i++)
        result->final_gradient_norm += grad[i] * grad[i];
    result->final_gradient_norm = sqrt(result->final_gradient_norm);

    free(params); free(grad); free(grad_old); free(direction);
    return 0;
}

/* ============================================================================
 * Part 4: Orthogonal Least Squares
 * Implements the ERR (Error Reduction Ratio) algorithm of
 * Chen, Billings & Luo (1989) for basis selection.
 * ============================================================================ */

int nlsid_optimize_ols(NARXModel* narx,
                        const NLSIDDataset* ds,
                        int max_bases,
                        double err_threshold,
                        NLSIDResult* result) {
    if (!narx || !ds || !result) return -1;
    if (!narx->expansion || narx->expansion->n_bases == 0) return -1;

    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;
    int N = ds->n_samples;
    int n_candidates = narx->expansion->n_bases;
    if (max_bases > n_candidates) max_bases = n_candidates;
    if (max_bases <= 0) return -1;

    int ny = narx->ny, nu = narx->nu, nk = narx->nk;
    int start_t = (ny > nu + nk) ? ny : nu + nk;
    int n_eff = N - start_t;
    if (n_eff <= 0) return -1;

    /* Build candidate regressor matrix Phi: n_eff x n_candidates */
    double* Phi = (double*)calloc((size_t)(n_eff * n_candidates), sizeof(double));
    for (int t = start_t; t < N; t++) {
        int row = t - start_t;
        int reg_dim;
        double* reg = nlsid_build_regressor(y, ny, u, nu, nk, t, &reg_dim);
        if (!reg) continue;
        for (int j = 0; j < n_candidates; j++) {
            Phi[row * n_candidates + j] =
                basis_evaluate(narx->expansion->bases[j], reg);
        }
        free(reg);
    }

    /* Output vector */
    double* Y = (double*)malloc((size_t)n_eff * sizeof(double));
    for (int t = 0; t < n_eff; t++) Y[t] = y[start_t + t];

    /* OLS: At each step, select candidate with largest ERR */
    int* selected = (int*)calloc((size_t)max_bases, sizeof(int));
    double* ERR = (double*)calloc((size_t)n_candidates, sizeof(double));
    double* W = (double*)malloc((size_t)n_eff * sizeof(double));
    double* G = (double*)calloc((size_t)(max_bases * n_eff), sizeof(double));
    double* weights = (double*)calloc((size_t)max_bases, sizeof(double));
    int n_selected = 0;
    double total_energy = 0.0;
    for (int t = 0; t < n_eff; t++) total_energy += Y[t] * Y[t];

    double residual_energy = total_energy;

    for (int step = 0; step < max_bases; step++) {
        /* Compute ERR for each unselected candidate */
        double best_err = -1.0;
        int best_idx = -1;

        for (int j = 0; j < n_candidates; j++) {
            /* Check if already selected */
            bool used = false;
            for (int s = 0; s < n_selected; s++)
                if (selected[s] == j) { used = true; break; }
            if (used) { ERR[j] = -1.0; continue; }

            /* Compute w_j = Phi_j^T Y_prev / (Phi_j^T Phi_j) */
            double num = 0.0, den = 0.0;
            for (int t = 0; t < n_eff; t++) {
                num += Phi[t * n_candidates + j] * Y[t];
                den += Phi[t * n_candidates + j] * Phi[t * n_candidates + j];
            }
            if (den < 1e-15) { ERR[j] = -1.0; continue; }
            double wj = num / den;
            double err_j = wj * wj * den / total_energy;
            ERR[j] = err_j;

            if (err_j > best_err) {
                best_err = err_j;
                best_idx = j;
            }
        }

        if (best_idx < 0 || best_err < err_threshold) break;

        selected[n_selected] = best_idx;

        /* Compute orthogonal regressor g_step = Phi_best */
        for (int t = 0; t < n_eff; t++)
            G[step * n_eff + t] = Phi[t * n_candidates + best_idx];

        /* Orthogonalize against previously selected */
        for (int s = 0; s < step; s++) {
            double dot_gs = 0.0, dot_gg = 0.0;
            for (int t = 0; t < n_eff; t++) {
                dot_gs += G[step * n_eff + t] * G[s * n_eff + t];
                dot_gg += G[s * n_eff + t] * G[s * n_eff + t];
            }
            double alpha_s = (dot_gg > 1e-15) ? dot_gs / dot_gg : 0.0;
            for (int t = 0; t < n_eff; t++)
                G[step * n_eff + t] -= alpha_s * G[s * n_eff + t];
        }

        /* Compute weight: g_step^T Y / (g_step^T g_step) */
        double num = 0.0, den = 0.0;
        for (int t = 0; t < n_eff; t++) {
            num += G[step * n_eff + t] * Y[t];
            den += G[step * n_eff + t] * G[step * n_eff + t];
        }
        weights[step] = (den > 1e-15) ? num / den : 0.0;

        /* Update residual */
        for (int t = 0; t < n_eff; t++)
            Y[t] -= weights[step] * G[step * n_eff + t];

        residual_energy = 0.0;
        for (int t = 0; t < n_eff; t++) residual_energy += Y[t] * Y[t];

        n_selected++;
    }

    /* Set selected weights in the NARX model */
    /* Zero all weights first */
    for (int i = 0; i < n_candidates; i++) narx->expansion->weights[i] = 0.0;
    narx->expansion->offset = 0.0;

    for (int s = 0; s < n_selected; s++) {
        int basis_idx = selected[s];
        narx->expansion->weights[basis_idx] = weights[s];
    }

    result->mse = residual_energy / (double)n_eff;
    result->converged = true;
    result->iterations_used = n_selected;

    free(Phi); free(Y); free(selected); free(ERR);
    free(W); free(G); free(weights);
    return 0;
}

/* ============================================================================
 * Part 5: Recursive Least Squares
 * ============================================================================ */

int nlsid_optimize_recursive_ls(NARXModel* narx,
                                 const NLSIDDataset* ds,
                                 double forgetting_factor,
                                 double* theta_final) {
    if (!narx || !ds || !theta_final) return -1;
    int n_params = narx->n_params;
    if (n_params <= 0) return -1;

    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;
    int N = ds->n_samples;

    double lambda = forgetting_factor;
    if (lambda <= 0.0 || lambda > 1.0) lambda = 0.99;

    /* P = (1/delta) * I, delta small */
    double delta = 1000.0;
    double* P = (double*)calloc((size_t)(n_params * n_params), sizeof(double));
    for (int i = 0; i < n_params; i++) P[i * n_params + i] = 1.0 / delta;

    double* theta = (double*)calloc((size_t)n_params, sizeof(double));
    int ny = narx->ny, nu = narx->nu, nk = narx->nk;

    for (int t = 1; t < N; t++) {
        int reg_dim;
        double* phi = nlsid_build_regressor(y, ny, u, nu, nk, t, &reg_dim);
        if (!phi) continue;

        /* phi^T * P */
        double* phiT_P = (double*)calloc((size_t)n_params, sizeof(double));
        for (int i = 0; i < n_params; i++)
            for (int j = 0; j < n_params; j++)
                phiT_P[i] += phi[j] * P[j * n_params + i];

        /* denom = lambda + phi^T P phi */
        double denom = lambda;
        for (int i = 0; i < n_params; i++) denom += phiT_P[i] * phi[i];

        /* K = P * phi / denom */
        double* K = (double*)calloc((size_t)n_params, sizeof(double));
        for (int i = 0; i < n_params; i++) {
            for (int j = 0; j < n_params; j++)
                K[i] += P[i * n_params + j] * phi[j];
            K[i] /= denom;
        }

        /* Prediction */
        double y_hat = 0.0;
        for (int i = 0; i < n_params; i++) y_hat += theta[i] * phi[i];
        double e_t = y[t] - y_hat;

        /* Update: theta = theta + K * e */
        for (int i = 0; i < n_params; i++) theta[i] += K[i] * e_t;

        /* Update P: P = (I - K phi^T) P / lambda */
        double* KP_phiT = (double*)calloc((size_t)(n_params * n_params), sizeof(double));
        for (int i = 0; i < n_params; i++)
            for (int j = 0; j < n_params; j++)
                KP_phiT[i * n_params + j] = K[i] * phiT_P[j];

        double* P_new = (double*)calloc((size_t)(n_params * n_params), sizeof(double));
        for (int i = 0; i < n_params; i++)
            for (int j = 0; j < n_params; j++) {
                P_new[i * n_params + j] = (P[i * n_params + j] - KP_phiT[i * n_params + j]) / lambda;
            }
        memcpy(P, P_new, (size_t)(n_params * n_params) * sizeof(double));

        free(phi); free(phiT_P); free(K); free(KP_phiT); free(P_new);
    }

    memcpy(theta_final, theta, (size_t)n_params * sizeof(double));
    free(P); free(theta);
    return 0;
}

/* ============================================================================
 * Part 6: Line Search and Initialization
 * ============================================================================ */

double nlsid_line_search_armijo(NLSIDModel* model,
                                 const NLSIDDataset* ds,
                                 const NLSIDConfig* config,
                                 const double* params_current,
                                 const double* direction,
                                 const double* gradient,
                                 double cost_current) {
    int n_params = model->n_params;
    double alpha = config->step_size_init;

    /* Compute directional derivative: ∇f^T d */
    double d_deriv = 0.0;
    for (int i = 0; i < n_params; i++) d_deriv += gradient[i] * direction[i];

    if (d_deriv >= 0.0) return 0.0; /* Not a descent direction */

    double* trial_params = (double*)malloc((size_t)n_params * sizeof(double));

    for (int iter = 0; iter < 50; iter++) {
        for (int i = 0; i < n_params; i++)
            trial_params[i] = params_current[i] + alpha * direction[i];
        model->set_params(model, trial_params);

        double cost_trial = nlsid_compute_cost(model, ds, config);

        /* Armijo condition: f(x + αd) ≤ f(x) + c₁ α ∇f^T d */
        if (cost_trial <= cost_current + config->armijo_c1 * alpha * d_deriv) {
            free(trial_params);
            model->set_params(model, params_current);
            return alpha;
        }
        alpha *= config->armijo_backtrack;
        if (alpha < config->step_size_min) break;
    }

    free(trial_params);
    model->set_params(model, params_current);
    return 0.0;
}

double nlsid_line_search_wolfe(NLSIDModel* model,
                                const NLSIDDataset* ds,
                                const NLSIDConfig* config,
                                const double* params_current,
                                const double* direction,
                                const double* gradient,
                                double cost_current) {
    /* Simplified: Armijo only, curvature condition skipped for efficiency */
    return nlsid_line_search_armijo(model, ds, config, params_current,
                                     direction, gradient, cost_current);
}

void nlsid_init_random(double* theta, int n_params, double sigma,
                        unsigned int* seed) {
    if (!theta || n_params <= 0) return;
    unsigned int s = seed ? *seed : 1;
    for (int i = 0; i < n_params; i++) {
        /* Box-Muller */
        double u1 = ((double)(s = s * 1103515245 + 12345) / 4294967296.0);
        double u2 = ((double)(s = s * 1103515245 + 12345) / 4294967296.0);
        if (u1 < 1e-10) u1 = 1e-10;
        theta[i] = sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }
    if (seed) *seed = s;
}

int nlsid_init_narx_ls(NARXModel* narx, const NLSIDDataset* ds) {
    if (!narx || !ds || !narx->expansion) return -1;
    const double* u = ds->input->channels[0]->data;
    const double* y = ds->output->channels[0]->data;
    int N = ds->n_samples;
    int n_params = narx->n_params;
    int ny = narx->ny, nu = narx->nu, nk = narx->nk;
    int start_t = (ny > nu + nk) ? ny : nu + nk;
    int n_eff = N - start_t;
    if (n_eff < n_params) return -1;

    /* Build Phi and solve normal equations */
    double* Phi = (double*)calloc((size_t)(n_eff * n_params), sizeof(double));
    double* Y = (double*)calloc((size_t)n_eff, sizeof(double));

    for (int t = start_t; t < N; t++) {
        int row = t - start_t;
        int reg_dim;
        double* reg = nlsid_build_regressor(y, ny, u, nu, nk, t, &reg_dim);
        if (!reg) continue;
        Phi[row * n_params] = 1.0; /* Bias */
        for (int j = 0; j < narx->expansion->n_bases && j + 1 < n_params; j++)
            Phi[row * n_params + j + 1] =
                basis_evaluate(narx->expansion->bases[j], reg);
        Y[row] = y[t];
        free(reg);
    }

    /* Normal equations: (Phi^T Phi) theta = Phi^T Y */
    double* PhiTPhi = (double*)calloc((size_t)(n_params * n_params), sizeof(double));
    double* PhiTY = (double*)calloc((size_t)n_params, sizeof(double));

    for (int i = 0; i < n_params; i++) {
        for (int t = 0; t < n_eff; t++)
            PhiTY[i] += Phi[t * n_params + i] * Y[t];
        for (int j = 0; j < n_params; j++)
            for (int t = 0; t < n_eff; t++)
                PhiTPhi[i * n_params + j] +=
                    Phi[t * n_params + i] * Phi[t * n_params + j];
    }

    nlsid_solve_linear_system(PhiTPhi, PhiTY, n_params, narx->theta);
    basis_expansion_unpack_params(narx->expansion, narx->theta);

    free(Phi); free(Y); free(PhiTPhi); free(PhiTY);
    return 0;
}

void nlsid_init_heuristic(double* theta, int n_params, unsigned int* seed) {
    if (!theta || n_params <= 0) return;
    unsigned int s = seed ? *seed : 1;
    for (int i = 0; i < n_params; i++) {
        s = s * 1103515245 + 12345;
        theta[i] = ((double)(s & 0xFFFF) / 65535.0 - 0.5) * 0.01;
    }
    if (seed) *seed = s;
}
/* ============================================================================
 * Part 7: Main Identification Interface
 * ============================================================================ */

int nlsid_identify(NLSIDModel* model,
                    const NLSIDDataset* estimation_data,
                    const NLSIDDataset* validation_data,
                    NLSIDConfig* config,
                    NLSIDResult* result) {
    if (!model || !estimation_data || !config || !result) return -1;

    /* Initialize result */
    memset(result, 0, sizeof(NLSIDResult));
    result->model = model;
    result->n_params = model->n_params;

    int n_params = model->n_params;
    double* params = (double*)malloc((size_t)n_params * sizeof(double));

    /* Parameter initialization */
    switch (config->init_method) {
        case 0: /* Zero init */
            memset(params, 0, (size_t)n_params * sizeof(double));
            break;
        case 1: /* Random init */
            { unsigned int seed = 42;
              nlsid_init_random(params, n_params, config->init_stddev, &seed); }
            break;
        case 2: /* Heuristic */
            { unsigned int seed = 42;
              nlsid_init_heuristic(params, n_params, &seed); }
            break;
        default:
            memset(params, 0, (size_t)n_params * sizeof(double));
    }
    model->set_params(model, params);

    /* Run optimization */
    int opt_status = -1;
    switch (config->algorithm) {
        case 0:
            opt_status = nlsid_optimize_gauss_newton(model, estimation_data,
                                                      config, result);
            break;
        case 1:
            opt_status = nlsid_optimize_levenberg_marquardt(model, estimation_data,
                                                             config, result);
            break;
        case 2:
            opt_status = nlsid_optimize_steepest_descent(model, estimation_data,
                                                          config, result);
            break;
        case 3:
            opt_status = nlsid_optimize_conjugate_gradient(model, estimation_data,
                                                            config, result);
            break;
        default:
            opt_status = nlsid_optimize_levenberg_marquardt(model, estimation_data,
                                                             config, result);
    }

    if (opt_status != 0) { free(params); return -1; }

    /* Get final parameters */
    model->get_params(model, params);
    result->param_estimates = (double*)malloc((size_t)n_params * sizeof(double));
    memcpy(result->param_estimates, params, (size_t)n_params * sizeof(double));

    /* Compute fit statistics on estimation data */
    const double* u_est = estimation_data->input->channels[0]->data;
    const double* y_est = estimation_data->output->channels[0]->data;
    int N_est = estimation_data->n_samples;

    double* y_hat_est = (double*)malloc((size_t)N_est * sizeof(double));
    for (int t = 0; t < N_est; t++) {
        double yh_est;
        model->predict_one_step(model, u_est, y_est, t, &yh_est);
        y_hat_est[t] = yh_est;
    }

    result->mse = nlsid_compute_mse(y_est, y_hat_est, N_est);
    result->fit_percent = nlsid_compute_fit(y_est, y_hat_est, N_est);

    /* Validate if validation data provided */
    if (validation_data && validation_data->n_samples > 0) {
        const double* u_val = validation_data->input->channels[0]->data;
        const double* y_val = validation_data->output->channels[0]->data;
        int N_val = validation_data->n_samples;

        double* y_hat_val = (double*)malloc((size_t)N_val * sizeof(double));
        for (int t = 0; t < N_val; t++) {
            double yh_val;
            model->predict_one_step(model, u_val, y_val, t, &yh_val);
            y_hat_val[t] = yh_val;
        }

        result->mse_validation = nlsid_compute_mse(y_val, y_hat_val, N_val);
        result->fit_percent_validation = nlsid_compute_fit(y_val, y_hat_val, N_val);
        free(y_hat_val);
    }

    /* Information criteria */
    double noise_var = result->mse;
    result->aic = nlsid_compute_aic(noise_var, N_est, n_params);
    result->bic = nlsid_compute_bic(noise_var, N_est, n_params);
    result->mdl = -N_est * log(noise_var) / 2.0 + n_params * log((double)N_est) / 2.0;
    result->fpe = nlsid_compute_fpe(noise_var, N_est, n_params);

    /* Residual statistics */
    double* residuals = (double*)malloc((size_t)N_est * sizeof(double));
    for (int t = 0; t < N_est; t++) residuals[t] = y_est[t] - y_hat_est[t];
    nlsid_residual_statistics(residuals, N_est,
        &result->residual_mean, &result->residual_variance,
        &result->residual_skewness, &result->residual_kurtosis);
    free(residuals);

    free(y_hat_est); free(params);
    return 0;
}

double nlsid_cross_validate(NLSIDModel* model,
                             const NLSIDDataset* ds,
                             NLSIDConfig* config,
                             int n_folds) {
    if (!model || !ds || n_folds < 2) return 0.0;
    int N = ds->n_samples;
    int fold_size = N / n_folds;
    if (fold_size < 10) return 0.0;

    double total_fit = 0.0;

    for (int fold = 0; fold < n_folds; fold++) {
        int val_start = fold * fold_size;
        int val_end = (fold == n_folds - 1) ? N : val_start + fold_size;
        int val_size = val_end - val_start;

        /* Create fold datasets (simplified: use indices) */
        NLSIDDataset* train_ds = nlsid_dataset_create(
            ds->input->n_channels, ds->output->n_channels,
            N - val_size, ds->sample_time);
        NLSIDDataset* val_ds = nlsid_dataset_create(
            ds->input->n_channels, ds->output->n_channels,
            val_size, ds->sample_time);

        /* Copy training data */
        int ti = 0, vi = 0;
        for (int t = 0; t < N; t++) {
            if (t >= val_start && t < val_end) {
                for (int ch = 0; ch < ds->input->n_channels; ch++)
                    val_ds->input->channels[ch]->data[vi] =
                        ds->input->channels[ch]->data[t];
                for (int ch = 0; ch < ds->output->n_channels; ch++)
                    val_ds->output->channels[ch]->data[vi] =
                        ds->output->channels[ch]->data[t];
                vi++;
            } else {
                for (int ch = 0; ch < ds->input->n_channels; ch++)
                    train_ds->input->channels[ch]->data[ti] =
                        ds->input->channels[ch]->data[t];
                for (int ch = 0; ch < ds->output->n_channels; ch++)
                    train_ds->output->channels[ch]->data[ti] =
                        ds->output->channels[ch]->data[t];
                ti++;
            }
        }

        NLSIDResult result;
        nlsid_identify(model, train_ds, val_ds, config, &result);
        total_fit += result.fit_percent_validation;

        nlsid_dataset_free(train_ds);
        nlsid_dataset_free(val_ds);
    }

    return total_fit / (double)n_folds;
}

/* ============================================================================
 * Part 8: Matrix Utility Functions
 * ============================================================================ */

int nlsid_solve_linear_system(double* A, double* b, int n, double* x) {
    if (!A || !b || !x || n <= 0) return -1;
    if (n == 1) {
        if (fabs(A[0]) < 1e-15) return -1;
        x[0] = b[0] / A[0];
        return 0;
    }

    /* Augmented matrix [A|b] */
    double* aug = (double*)malloc((size_t)(n * (n + 1)) * sizeof(double));
    if (!aug) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * (n + 1) + j] = A[i * n + j];
        aug[i * (n + 1) + n] = b[i];
    }

    /* Gaussian elimination with partial pivoting */
    for (int col = 0; col < n; col++) {
        /* Find pivot */
        int pivot_row = col;
        double max_val = fabs(aug[col * (n + 1) + col]);
        for (int row = col + 1; row < n; row++) {
            double val = fabs(aug[row * (n + 1) + col]);
            if (val > max_val) { max_val = val; pivot_row = row; }
        }
        if (max_val < 1e-15) { free(aug); return -1; }

        /* Swap rows */
        if (pivot_row != col) {
            for (int j = 0; j <= n; j++) {
                double tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[pivot_row * (n + 1) + j];
                aug[pivot_row * (n + 1) + j] = tmp;
            }
        }

        /* Eliminate below */
        double pivot = aug[col * (n + 1) + col];
        for (int row = col + 1; row < n; row++) {
            double factor = aug[row * (n + 1) + col] / pivot;
            for (int j = col; j <= n; j++)
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
        }
    }

    /* Back substitution */
    for (int i = n - 1; i >= 0; i--) {
        x[i] = aug[i * (n + 1) + n];
        for (int j = i + 1; j < n; j++)
            x[i] -= aug[i * (n + 1) + j] * x[j];
        double diag = aug[i * (n + 1) + i];
        if (fabs(diag) < 1e-15) { free(aug); return -1; }
        x[i] /= diag;
    }

    free(aug);
    return 0;
}

int nlsid_matrix_inverse(double* A, int n, double* A_inv) {
    if (!A || !A_inv || n <= 0) return -1;
    /* Augment with identity: [A|I] → [I|A^{-1}] */
    double* aug = (double*)calloc((size_t)(n * (2 * n)), sizeof(double));
    if (!aug) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * 2*n + j] = A[i * n + j];
        aug[i * 2*n + n + i] = 1.0;
    }

    for (int col = 0; col < n; col++) {
        int pivot = col;
        double pmax = fabs(aug[col * 2*n + col]);
        for (int r = col + 1; r < n; r++) {
            if (fabs(aug[r * 2*n + col]) > pmax) {
                pmax = fabs(aug[r * 2*n + col]); pivot = r;
            }
        }
        if (pmax < 1e-15) { free(aug); return -1; }
        if (pivot != col) {
            for (int j = 0; j < 2*n; j++) {
                double t = aug[col * 2*n + j];
                aug[col * 2*n + j] = aug[pivot * 2*n + j];
                aug[pivot * 2*n + j] = t;
            }
        }
        double piv = aug[col * 2*n + col];
        for (int j = 0; j < 2*n; j++) aug[col * 2*n + j] /= piv;
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            double factor = aug[r * 2*n + col];
            for (int j = 0; j < 2*n; j++)
                aug[r * 2*n + j] -= factor * aug[col * 2*n + j];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_inv[i * n + j] = aug[i * 2*n + n + j];
    free(aug);
    return 0;
}

int nlsid_matrix_condition(const double* A, int n, double* condition) {
    if (!A || !condition || n <= 0) return -1;
    /* Frobenius norm based condition estimate */
    double norm_A = 0.0;
    for (int i = 0; i < n*n; i++) norm_A += A[i] * A[i];
    norm_A = sqrt(norm_A);

    double* A_inv = (double*)malloc((size_t)(n*n) * sizeof(double));
    if (nlsid_matrix_inverse((double*)A, n, A_inv) != 0) {
        *condition = 1e16;
        free(A_inv);
        return -1;
    }

    double norm_inv = 0.0;
    for (int i = 0; i < n*n; i++) norm_inv += A_inv[i] * A_inv[i];
    norm_inv = sqrt(norm_inv);

    *condition = norm_A * norm_inv;
    free(A_inv);
    return 0;
}

void nlsid_matrix_mult(const double* A, const double* x, int n, int m,
                        double* y) {
    if (!A || !x || !y) return;
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < m; j++)
            y[i] += A[i * m + j] * x[j];
    }
}

void nlsid_matrix_transpose_mult(const double* A, const double* x, int n, int m,
                                  double* y) {
    if (!A || !x || !y) return;
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++)
            y[i] += A[j * m + i] * x[j];
    }
}

void nlsid_matrix_matmul(const double* A, const double* B, int m, int k, int n,
                          double* C) {
    if (!A || !B || !C) return;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int p = 0; p < k; p++)
                C[i * n + j] += A[i * k + p] * B[p * n + j];
        }
    }
}

int nlsid_cholesky(const double* A, int n, double* L) {
    if (!A || !L || n <= 0) return -1;
    memset(L, 0, (size_t)(n*n) * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            if (j == i) {
                for (int k = 0; k < j; k++)
                    sum += L[j * n + k] * L[j * n + k];
                double diag = A[j * n + j] - sum;
                if (diag <= 1e-15) return -1;
                L[j * n + j] = sqrt(diag);
            } else {
                for (int k = 0; k < j; k++)
                    sum += L[i * n + k] * L[j * n + k];
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return 0;
}

void nlsid_cholesky_solve(const double* L, const double* b, int n, double* x) {
    if (!L || !b || !x || n <= 0) return;
    double* y = (double*)calloc((size_t)n, sizeof(double));

    /* Forward substitution: L y = b */
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i * n + j] * y[j];
        y[i] /= L[i * n + i];
    }

    /* Back substitution: L^T x = y */
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= L[j * n + i] * x[j];
        x[i] /= L[i * n + i];
    }
    free(y);
}
