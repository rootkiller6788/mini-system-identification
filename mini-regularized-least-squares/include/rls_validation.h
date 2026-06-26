#ifndef RLS_VALIDATION_H
#define RLS_VALIDATION_H

#include "rls_core.h"

/* ============================================================================
 * Regularization Parameter Selection and Model Validation (L5/L6)
 *
 * Methods for choosing the optimal regularization parameter lambda:
 *
 * 1. K-fold Cross-Validation (CV):
 *    Partition data into K folds. For each lambda:
 *      For each fold i: fit on K-1 folds, validate on fold i.
 *    Choose lambda minimizing average validation error.
 *
 * 2. Generalized Cross-Validation (GCV):
 *    GCV(lambda) = ||y - y_hat||^2 / (n - df(lambda))^2
 *    where df(lambda) = effective degrees of freedom.
 *    Ref: Golub, Heath, Wahba (1979) Technometrics 21(2)
 *
 * 3. L-curve Criterion:
 *    Plot log||Phi*theta - y|| vs log||theta|| for different lambda.
 *    Optimal lambda at point of maximum curvature (corner).
 *    Ref: Hansen (1992) "Analysis of Discrete Ill-Posed Problems"
 *
 * 4. AICc (Corrected Akaike Information Criterion):
 *    AICc = n*log(RSS/n) + 2*df + 2*df*(df+1)/(n-df-1)
 *    Ref: Hurvich & Tsai (1989) Biometrika 76(2)
 *
 * 5. SURE (Stein's Unbiased Risk Estimate):
 *    SURE(lambda) = ||y - y_hat||^2 + 2*sigma^2*df(lambda) - n*sigma^2
 *    Unbiased estimate of prediction risk when sigma^2 is known.
 *
 * 6. Empirical Bayes / Marginal Likelihood:
 *    lambda_opt = argmax p(y|lambda) under Gaussian prior.
 *    Ref: Pillonetto & De Nicolao (2010)
 *
 * 7. BIC (Bayesian Information Criterion):
 *    BIC = n*log(RSS/n) + df*log(n)
 * ============================================================================ */

/* ---------- K-Fold Cross-Validation ---------- */

/** Perform K-fold CV over a grid of lambda values.
 *  Returns optimal lambda and fills cv_scores array.
 *  On input, sel->lambda_grid should contain n_lambda values.
 *  On output, sel->cv_scores filled, sel->lambda_opt set. */
int  rls_kfold_cv(const RLSMatrix *Phi, const RLSVector *y,
                  RLSLambdaSelection *sel, RLSRegType reg_type,
                  double alpha, const RLSOptions *opt);

/** Compute prediction error for a single fold. Internal use. */
double rls_kfold_single_fold(const RLSMatrix *Phi_train, const RLSVector *y_train,
                              const RLSMatrix *Phi_val, const RLSVector *y_val,
                              double lambda, RLSRegType reg_type, double alpha,
                              const RLSOptions *opt);

/* ---------- Generalized Cross-Validation ---------- */

/** Compute GCV score for a given lambda.
 *  GCV(lambda) = RSS / (n - df(lambda))^2 * n
 *  where df(lambda) = sum_i s_i^2 / (s_i^2 + lambda) for ridge (s_i = singular values). */
double rls_gcv_score(const RLSMatrix *Phi, const RLSVector *y,
                     double lambda, RLSRegType reg_type, double alpha,
                     const RLSOptions *opt);

/** Find optimal lambda via GCV minimization over a grid. */
int  rls_gcv_optimize(const RLSMatrix *Phi, const RLSVector *y,
                      RLSLambdaSelection *sel, RLSRegType reg_type,
                      double alpha, const RLSOptions *opt);

/* ---------- L-Curve ---------- */

/** Compute L-curve coordinates: (log||residual||, log||theta||)
 *  for a grid of lambda values. */
void rls_lcurve_compute(const RLSMatrix *Phi, const RLSVector *y,
                        const double *lambdas, int n_lambdas,
                        RLSRegType reg_type, double alpha,
                        const RLSOptions *opt,
                        double *log_rho, double *log_eta);

/** Find L-curve corner (point of maximum curvature).
 *  curvature = 2 * (rho'*eta'' - rho''*eta') / (rho'^2 + eta'^2)^(3/2) */
double rls_lcurve_corner(const double *log_rho, const double *log_eta,
                          int n, const double *lambdas);

/** Full L-curve lambda selection */
int  rls_lcurve_optimize(const RLSMatrix *Phi, const RLSVector *y,
                         RLSLambdaSelection *sel, RLSRegType reg_type,
                         double alpha, const RLSOptions *opt);

/* ---------- Information Criteria ---------- */

/** Compute AICc for given lambda.
 *  AICc = n*log(RSS/n) + 2*df + 2*df*(df+1)/(n-df-1) */
double rls_aicc_score(const RLSMatrix *Phi, const RLSVector *y,
                      double lambda, RLSRegType reg_type, double alpha,
                      const RLSOptions *opt);

/** Compute BIC for given lambda.
 *  BIC = n*log(RSS/n) + df*log(n) */
double rls_bic_score(const RLSMatrix *Phi, const RLSVector *y,
                     double lambda, RLSRegType reg_type, double alpha,
                     const RLSOptions *opt);

/** Select lambda via AICc minimization */
int  rls_aicc_optimize(const RLSMatrix *Phi, const RLSVector *y,
                       RLSLambdaSelection *sel, RLSRegType reg_type,
                       double alpha, const RLSOptions *opt);

/* ---------- SURE ---------- */

/** Compute SURE criterion for given lambda and known noise variance sigma2. */
double rls_sure_score(const RLSMatrix *Phi, const RLSVector *y,
                      double lambda, double sigma2, const RLSOptions *opt);

int  rls_sure_optimize(const RLSMatrix *Phi, const RLSVector *y,
                       double sigma2, RLSLambdaSelection *sel,
                       const RLSOptions *opt);

/* ---------- Empirical Bayes / Marginal Likelihood ---------- */

/** Marginal likelihood for ridge: log p(y|lambda) = -0.5*(y^T (K+lambda*I)^{-1} y + log det(K+lambda*I) + n*log(2*pi))
 *  where K = Phi * Phi^T. Maximized to select lambda. */
double rls_marginal_likelihood(const RLSMatrix *Phi, const RLSVector *y,
                                double lambda);

int  rls_empirical_bayes_optimize(const RLSMatrix *Phi, const RLSVector *y,
                                   RLSLambdaSelection *sel,
                                   const RLSOptions *opt);

/* ---------- Unified Lambda Selection ---------- */

/** Auto-select lambda using the specified method.
 *  Fills sel->lambda_opt and sel->lambda_opt_score.
 *  Returns 0 on success. */
int  rls_select_lambda(const RLSMatrix *Phi, const RLSVector *y,
                       RLSLambdaSelection *sel, RLSRegType reg_type,
                       double alpha, const RLSOptions *opt);

/** Generate a default lambda grid (log-spaced from lambda_max to lambda_min). */
void rls_generate_lambda_grid(RLSLambdaSelection *sel);

#endif
