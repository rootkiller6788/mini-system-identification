#include "uq_bayesian.h"
#include "uq_sampling.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Example 2: Bayesian Parameter Estimation via MCMC
 * Demonstrates Metropolis-Hastings sampling for a simple normal model,
 * adaptive Metropolis, convergence diagnostics, and posterior summaries.
 */

int main(void) {
    printf("=== MCMC Bayesian Parameter Estimation ===\n\n");

    /* Generate synthetic data from N(μ=5.0, σ=2.0) */
    int n_data = 50;
    double true_mu = 5.0, true_sigma = 2.0;
    double* y_obs = (double*)malloc(n_data * sizeof(double));

    UQDistribution* truth = uq_dist_create_normal(true_mu, true_sigma);
    uq_dist_sample_n(truth, n_data, y_obs);

    printf("True parameters: μ=%.1f, σ=%.1f (n=%d observations)\n\n",
           true_mu, true_sigma, n_data);

    /* Set up Bayesian model */
    /* Prior: μ ~ N(0, 10²) — weakly informative */
    UQPriorDistribution* prior_mu = uq_prior_create(
        uq_dist_create_normal(0.0, 10.0), true);

    /* Likelihood: y_i ~ N(μ, σ=2.0) (σ known for simplicity) */
    UQLikelihood* lh = uq_likelihood_create(UQ_LIKELIHOOD_NORMAL, n_data);
    lh->obs_ptr = y_obs;
    lh->params.sigma = 2.0;

    /* Prior predictive check */
    printf("Prior predictive: μ ~ N(0, 10²), likelihood ~ N(μ, 2²)\n");
    double prior_pred[10];
    double* prior_samples = (double*)malloc(10 * sizeof(double));
    uq_dist_sample_n(prior_mu->dist, 10, prior_samples);
    for (int i = 0; i < 10 && i < 10; i++)
        printf("  μ_sample[%d] = %.3f\n", i, prior_samples[i]);

    /* MCMC: 3 chains, 2000 iterations each */
    printf("\nRunning MCMC (3 chains × 2000 iterations)...\n");
    UQMCMCState* mcmc = uq_mcmc_create(3, 2000, 1, NULL);

    /* Override default initial values with dispersed starts */
    double init1[] = {-3.0}, init2[] = {0.0}, init3[] = {3.0};
    uq_mcmc_set_initial(mcmc, init1, 0);
    uq_mcmc_set_initial(mcmc, init2, 1);
    uq_mcmc_set_initial(mcmc, init3, 2);

    /* Run Metropolis-Hastings */
    double* pred_buffer = (double*)malloc(n_data * sizeof(double));
    uq_mh_sample(mcmc, prior_mu, lh, pred_buffer, 3);

    printf("\nMCMC Results:\n");
    printf("  Chain 0 acceptance rate: %.3f\n", uq_mcmc_acceptance_rate(mcmc, 0));
    printf("  Chain 1 acceptance rate: %.3f\n", uq_mcmc_acceptance_rate(mcmc, 1));
    printf("  Chain 2 acceptance rate: %.3f\n", uq_mcmc_acceptance_rate(mcmc, 2));

    /* Posterior summary from chains (post burn-in) */
    int burn = mcmc->burn_in;
    int n_post = 2000 - burn;
    double post_sum = 0.0, post_s2 = 0.0;
    int count = 0;

    for (int c = 0; c < 3; c++)
        for (int iter = burn; iter < 2000; iter++) {
            double val = mcmc->chains[c][iter]; /* 1 param */
            post_sum += val; post_s2 += val * val; count++;
        }

    double post_mean = post_sum / count;
    double post_var = post_s2 / count - post_mean * post_mean;
    double post_sd = sqrt(post_var);

    printf("\nPosterior Summary (after burn-in):\n");
    printf("  E[μ|y] = %.4f  (true: %.1f)\n", post_mean, true_mu);
    printf("  SD[μ|y] = %.4f\n", post_sd);
    printf("  95%% Credible Interval: [%.4f, %.4f]\n",
           post_mean - 1.96 * post_sd, post_mean + 1.96 * post_sd);

    /* Geweke convergence diagnostic */
    UQConvergenceDiagnostic* diag = uq_conv_create(20);
    uq_mcmc_diagnostics(mcmc, diag);
    printf("  Geweke Z = %.4f (%s)\n", diag->geweke_z,
           diag->converged ? "converged" : "NOT converged");
    printf("  Gelman-Rubin R-hat = %.4f\n", diag->gelman_rubin_rhat);

    /* Effective sample size */
    double* chain0 = (double*)malloc(2000 * sizeof(double));
    uq_mcmc_trace(chain0, mcmc, 0, 0);
    int ess = uq_conv_effective_sample_size(diag, chain0, 2000);
    printf("  Effective Sample Size (chain 0) = %d\n", ess);
    printf("  Autocorrelation at lag 1: %.4f\n", diag->autocorrelation[1]);

    /* Now demonstrate Adaptive Metropolis */
    printf("\n--- Adaptive Metropolis ---\n");
    UQMCMCState* am = uq_mcmc_create(1, 1000, 1, NULL);
    uq_am_sample(am, prior_mu, lh, pred_buffer);
    printf("  AM acceptance rate: %.3f\n", uq_mcmc_acceptance_rate(am, 0));

    /* Bootstrap comparison */
    printf("\n--- Bootstrap Comparison ---\n");
    UQBootstrap* bs = uq_bootstrap_create(UQ_BOOTSTRAP_STANDARD, n_data, 1000, 1);
    uq_bootstrap_set_data(bs, y_obs);
    uq_bootstrap_generate_replicates(bs);

    /* Compute mean statistic */
    double bs_mean_sum = 0.0, bs_s2 = 0.0;
    for (int b = 0; b < 1000; b++) {
        double* rep = bs->bootstrap_replicates[b];
        double m = 0.0;
        for (int i = 0; i < n_data; i++) m += rep[i];
        m /= n_data;
        bs_mean_sum += m;
        bs_s2 += m * m;
    }
    double bs_mu = bs_mean_sum / 1000;
    double bs_se = sqrt(bs_s2 / 1000 - bs_mu * bs_mu);
    printf("  Bootstrap mean of μ̂ = %.4f (SE = %.4f)\n", bs_mu, bs_se);
    printf("  Analytical SE = %.4f\n", true_sigma / sqrt((double)n_data));

    /* Cleanup */
    free(y_obs);
    free(prior_samples);
    free(pred_buffer);
    free(chain0);
    uq_dist_free(truth);
    uq_prior_free(prior_mu);
    uq_likelihood_free(lh);
    uq_mcmc_free(mcmc);
    uq_mcmc_free(am);
    uq_conv_free(diag);
    uq_bootstrap_free(bs);

    printf("\n=== Example 2 Complete ===\n");
    return 0;
}
