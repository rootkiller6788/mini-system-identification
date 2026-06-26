#include "uq_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define UQ_PI   3.14159265358979323846
#define UQ_EPS  1e-12
#define UQ_QMZ  0.91893853320467274178  /* 0.5 * ln(2π) */

/* ============================================================================
 * Internal Helpers — Random Number Generation
 * ============================================================================ */

static double urand(void) {
    return ((double)rand() / ((double)RAND_MAX + 1.0));
}

static double gauss_rand(void) {
    double u1, u2;
    do { u1 = urand(); } while (u1 < 1e-15);
    u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * UQ_PI * u2);
}

static double gauss_pdf(double x, double mu, double sigma) {
    double z = (x - mu) / sigma;
    return exp(-0.5 * z * z) / (sigma * sqrt(2.0 * UQ_PI));
}

static double gauss_cdf_approx(double x) {
    /* Abramowitz & Stegun 26.2.17 approximation */
    double t = 1.0 / (1.0 + 0.2316419 * fabs(x));
    double b[] = {0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429};
    double poly = t * (b[0] + t * (b[1] + t * (b[2] + t * (b[3] + t * b[4]))));
    double phi = 1.0 - (1.0 / sqrt(2.0 * UQ_PI)) * exp(-0.5 * x * x) * poly;
    return (x >= 0.0) ? phi : 1.0 - phi;
}

static double student_t_pdf(double x, double df) {
    double norm = exp(uq_stats_lgamma((df + 1.0) * 0.5) - uq_stats_lgamma(df * 0.5))
                  / sqrt(df * UQ_PI);
    return norm * pow(1.0 + x * x / df, -(df + 1.0) * 0.5);
}

static double chi2_pdf(double x, double df) {
    if (x <= 0.0) return 0.0;
    double k = df * 0.5;
    return pow(x, k - 1.0) * exp(-x * 0.5) / (pow(2.0, k) * exp(uq_stats_lgamma(k)));
}

/* ============================================================================
 * Distribution Creation
 * ============================================================================ */

UQDistribution* uq_dist_create_normal(double mean, double std) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_NORMAL;
    d->n_params = 2;
    d->params = (double*)malloc(2 * sizeof(double));
    d->params[0] = mean;
    d->params[1] = std;
    d->lower_bound = -1e308;
    d->upper_bound = 1e308;
    d->is_bounded = false;
    d->mean = mean;
    d->variance = std * std;
    d->median = mean;
    d->mode = mean;
    d->entropy = 0.5 * log(2.0 * UQ_PI * exp(1.0) * std * std);
    return d;
}

UQDistribution* uq_dist_create_uniform(double a, double b) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_UNIFORM;
    d->n_params = 2;
    d->params = (double*)malloc(2 * sizeof(double));
    d->params[0] = a;
    d->params[1] = b;
    d->lower_bound = a;
    d->upper_bound = b;
    d->is_bounded = true;
    d->mean = 0.5 * (a + b);
    d->variance = (b - a) * (b - a) / 12.0;
    d->median = d->mean;
    d->mode = d->mean;
    d->entropy = log(b - a);
    return d;
}

UQDistribution* uq_dist_create_student_t(double location, double scale, double df) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_STUDENT_T;
    d->n_params = 3;
    d->params = (double*)malloc(3 * sizeof(double));
    d->params[0] = location;
    d->params[1] = scale;
    d->params[2] = df;
    d->lower_bound = -1e308;
    d->upper_bound = 1e308;
    d->is_bounded = false;
    d->mean = (df > 1.0) ? location : NAN;
    d->variance = (df > 2.0) ? scale * scale * df / (df - 2.0) : NAN;
    d->median = location;
    d->mode = location;
    return d;
}

UQDistribution* uq_dist_create_chi2(double df) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_CHI2;
    d->n_params = 1;
    d->params = (double*)malloc(1 * sizeof(double));
    d->params[0] = df;
    d->lower_bound = 0.0;
    d->upper_bound = 1e308;
    d->is_bounded = true;
    d->mean = df;
    d->variance = 2.0 * df;
    d->median = df * pow(1.0 - 2.0 / (9.0 * df), 3.0);
    d->mode = (df >= 2.0) ? df - 2.0 : 0.0;
    return d;
}

UQDistribution* uq_dist_create_lognormal(double mu, double sigma) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_LOG_NORMAL;
    d->n_params = 2;
    d->params = (double*)malloc(2 * sizeof(double));
    d->params[0] = mu;
    d->params[1] = sigma;
    d->lower_bound = 0.0;
    d->upper_bound = 1e308;
    d->is_bounded = true;
    double s2 = sigma * sigma;
    d->mean = exp(mu + s2 * 0.5);
    d->variance = (exp(s2) - 1.0) * exp(2.0 * mu + s2);
    d->median = exp(mu);
    d->mode = exp(mu - s2);
    d->entropy = mu + 0.5 + log(sigma * sqrt(2.0 * UQ_PI));
    return d;
}

UQDistribution* uq_dist_create_beta(double alpha, double beta) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_BETA;
    d->n_params = 2;
    d->params = (double*)malloc(2 * sizeof(double));
    d->params[0] = alpha;
    d->params[1] = beta;
    d->lower_bound = 0.0;
    d->upper_bound = 1.0;
    d->is_bounded = true;
    double ab = alpha + beta;
    d->mean = alpha / ab;
    d->variance = alpha * beta / (ab * ab * (ab + 1.0));
    d->mode = (alpha > 1.0 && beta > 1.0) ? (alpha - 1.0) / (ab - 2.0) : NAN;
    return d;
}

UQDistribution* uq_dist_create_gamma(double shape, double scale) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_GAMMA;
    d->n_params = 2;
    d->params = (double*)malloc(2 * sizeof(double));
    d->params[0] = shape;
    d->params[1] = scale;
    d->lower_bound = 0.0;
    d->upper_bound = 1e308;
    d->is_bounded = true;
    d->mean = shape * scale;
    d->variance = shape * scale * scale;
    d->mode = (shape >= 1.0) ? (shape - 1.0) * scale : 0.0;
    d->entropy = shape + log(scale) + uq_stats_lgamma(shape)
                 + (1.0 - shape) * uq_stats_digamma(shape);
    return d;
}

UQDistribution* uq_dist_create_multivariate_normal(int dim, double* mean, UQMatrix* cov) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_MULTIVARIATE_NORMAL;
    d->dimension = dim;
    d->n_params = dim + dim * dim;
    d->params = (double*)malloc((dim + dim * dim) * sizeof(double));
    memcpy(d->params, mean, dim * sizeof(double));
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            d->params[dim + i * dim + j] = uq_matrix_get(cov, i, j);
    d->lower_bound = -1e308;
    d->upper_bound = 1e308;
    d->is_bounded = false;
    d->mean = mean[0];  /* first component */
    d->variance = uq_matrix_get(cov, 0, 0);
    d->entropy = 0.5 * log(pow(2.0 * UQ_PI * exp(1.0), dim)
                 * fabs(uq_matrix_determinant(cov)));
    return d;
}

UQDistribution* uq_dist_create_empirical(double* samples, int n) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_EMPIRICAL;
    d->n_samples = n;
    d->sample_capacity = n;
    d->samples = (double*)malloc(n * sizeof(double));
    memcpy(d->samples, samples, n * sizeof(double));
    /* Compute summary */
    double sum = 0.0, s2 = 0.0;
    for (int i = 0; i < n; i++) sum += samples[i];
    d->mean = sum / n;
    for (int i = 0; i < n; i++) s2 += (samples[i] - d->mean) * (samples[i] - d->mean);
    d->variance = s2 / (n - 1.0);
    d->median = uq_dist_quantile(d, 0.5);
    return d;
}

UQDistribution* uq_dist_create_kde(double* samples, int n, double bandwidth) {
    UQDistribution* d = (UQDistribution*)calloc(1, sizeof(UQDistribution));
    d->type = UQ_DIST_KERNEL_DENSITY;
    d->n_samples = n;
    d->sample_capacity = n;
    d->samples = (double*)malloc(n * sizeof(double));
    memcpy(d->samples, samples, n * sizeof(double));
    d->n_params = 1;
    d->params = (double*)malloc(sizeof(double));
    /* Scott's rule if bandwidth not specified */
    if (bandwidth <= 0.0) {
        double std_dev = sqrt(d->variance);
        d->params[0] = 1.06 * std_dev * pow((double)n, -0.2);
    } else {
        d->params[0] = bandwidth;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += samples[i];
    d->mean = sum / n;
    return d;
}

void uq_dist_free(UQDistribution* dist) {
    if (!dist) return;
    free(dist->params);
    free(dist->samples);
    free(dist->gp_training_x);
    free(dist->gp_training_y);
    free(dist);
}

/* ============================================================================
 * Distribution PDF/CDF/Quantile
 * ============================================================================ */

double uq_dist_pdf(UQDistribution* dist, double x) {
    if (!dist) return NAN;
    switch (dist->type) {
    case UQ_DIST_NORMAL:
        return gauss_pdf(x, dist->params[0], dist->params[1]);
    case UQ_DIST_UNIFORM:
        if (x < dist->params[0] || x > dist->params[1]) return 0.0;
        return 1.0 / (dist->params[1] - dist->params[0]);
    case UQ_DIST_STUDENT_T:
        return student_t_pdf((x - dist->params[0]) / dist->params[1], dist->params[2])
               / dist->params[1];
    case UQ_DIST_CHI2:
        return chi2_pdf(x, dist->params[0]);
    case UQ_DIST_LOG_NORMAL:
        if (x <= 0.0) return 0.0;
        return gauss_pdf(log(x), dist->params[0], dist->params[1]) / x;
    case UQ_DIST_BETA:
        if (x <= 0.0 || x >= 1.0) return 0.0;
        return pow(x, dist->params[0] - 1.0) * pow(1.0 - x, dist->params[1] - 1.0)
               / exp(uq_stats_lgamma(dist->params[0]) + uq_stats_lgamma(dist->params[1])
               - uq_stats_lgamma(dist->params[0] + dist->params[1]));
    case UQ_DIST_GAMMA:
        if (x <= 0.0) return 0.0;
        return pow(x, dist->params[0] - 1.0) * exp(-x / dist->params[1])
               / (pow(dist->params[1], dist->params[0]) * exp(uq_stats_lgamma(dist->params[0])));
    case UQ_DIST_EXPONENTIAL:
        if (x < 0.0) return 0.0;
        return dist->params[0] * exp(-dist->params[0] * x);
    case UQ_DIST_EMPIRICAL:
    case UQ_DIST_KERNEL_DENSITY: {
        /* KDE with Gaussian kernel */
        if (!dist->samples || dist->n_samples == 0) return 0.0;
        double bw = dist->params[0];
        double sum = 0.0;
        for (int i = 0; i < dist->n_samples; i++) {
            double z = (x - dist->samples[i]) / bw;
            sum += exp(-0.5 * z * z);
        }
        return sum / (dist->n_samples * bw * sqrt(2.0 * UQ_PI));
    }
    case UQ_DIST_CAUCHY:
        return 1.0 / (UQ_PI * dist->params[1] * (1.0
               + ((x - dist->params[0]) / dist->params[1])
               * ((x - dist->params[0]) / dist->params[1])));
    default:
        return NAN;
    }
}

double uq_dist_cdf(UQDistribution* dist, double x) {
    if (!dist) return NAN;
    switch (dist->type) {
    case UQ_DIST_NORMAL:
        return gauss_cdf_approx((x - dist->params[0]) / dist->params[1]);
    case UQ_DIST_UNIFORM:
        if (x <= dist->params[0]) return 0.0;
        if (x >= dist->params[1]) return 1.0;
        return (x - dist->params[0]) / (dist->params[1] - dist->params[0]);
    case UQ_DIST_CHI2: {
        if (x <= 0.0) return 0.0;
        return uq_stats_beta_regularized(dist->params[0] * 0.5, 0.5, 0.0);
        /* Note: chi2 CDF via incomplete gamma — use quadrature approximation */
        double sum = 0.0, term = 1.0, k = dist->params[0] * 0.5;
        for (int j = 0; j < 200; j++) {
            term *= x * 0.5 / (k + (double)j);
            sum += term;
            if (term < 1e-15) break;
        }
        return sum * exp(-x * 0.5 + k * log(x * 0.5) - uq_stats_lgamma(k));
    }
    case UQ_DIST_STUDENT_T: {
        double t = (x - dist->params[0]) / dist->params[1];
        double df = dist->params[2];
        /* Use regularized incomplete beta for symmetric t-CDF */
        double xb = df / (df + t * t);
        double ib = uq_stats_beta_regularized(df * 0.5, 0.5, xb);
        return 0.5 * (1.0 + (t >= 0 ? 1.0 - ib : ib - 1.0));
    }
    case UQ_DIST_LOG_NORMAL:
        if (x <= 0.0) return 0.0;
        return gauss_cdf_approx((log(x) - dist->params[0]) / dist->params[1]);
    case UQ_DIST_BETA:
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        return uq_stats_beta_regularized(dist->params[0], dist->params[1], x);
    case UQ_DIST_GAMMA:
        if (x <= 0.0) return 0.0;
        return uq_stats_beta_regularized(dist->params[0], 1.0, 0.0);
        /* use gamma CDF via incomplete gamma — truncated series */
    case UQ_DIST_EXPONENTIAL:
        if (x <= 0.0) return 0.0;
        return 1.0 - exp(-dist->params[0] * x);
    case UQ_DIST_CAUCHY:
        return 0.5 + atan((x - dist->params[0]) / dist->params[1]) / UQ_PI;
    case UQ_DIST_EMPIRICAL: {
        if (!dist->samples || dist->n_samples == 0) return NAN;
        int count = 0;
        for (int i = 0; i < dist->n_samples; i++)
            if (dist->samples[i] <= x) count++;
        return (double)count / (double)dist->n_samples;
    }
    default:
        return NAN;
    }
}

double uq_dist_quantile(UQDistribution* dist, double p) {
    if (!dist || p < 0.0 || p > 1.0) return NAN;
    if (p == 0.0) return dist->lower_bound;
    if (p == 1.0) return dist->upper_bound;

    switch (dist->type) {
    case UQ_DIST_NORMAL:
        return dist->params[0] + dist->params[1] * uq_stats_normal_quantile(p);
    case UQ_DIST_UNIFORM:
        return dist->params[0] + p * (dist->params[1] - dist->params[0]);
    case UQ_DIST_LOG_NORMAL:
        return exp(dist->params[0] + dist->params[1] * uq_stats_normal_quantile(p));
    case UQ_DIST_EXPONENTIAL:
        return -log(1.0 - p) / dist->params[0];
    case UQ_DIST_CAUCHY:
        return dist->params[0] + dist->params[1] * tan(UQ_PI * (p - 0.5));
    case UQ_DIST_EMPIRICAL: {
        if (!dist->samples || dist->n_samples == 0) return NAN;
        /* Sort and interpolate */
        double* sorted = (double*)malloc(dist->n_samples * sizeof(double));
        memcpy(sorted, dist->samples, dist->n_samples * sizeof(double));
        /* Simple sort */
        for (int i = 0; i < dist->n_samples - 1; i++)
            for (int j = i + 1; j < dist->n_samples; j++)
                if (sorted[i] > sorted[j]) {
                    double tmp = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = tmp;
                }
        double pos = p * (dist->n_samples - 1.0);
        int lo = (int)floor(pos);
        int hi = (lo + 1 < dist->n_samples) ? lo + 1 : lo;
        double frac = pos - (double)lo;
        double result = sorted[lo] + frac * (sorted[hi] - sorted[lo]);
        free(sorted);
        return result;
    }
    default: {
        /* Bisection search using CDF */
        double lo = dist->lower_bound, hi = dist->upper_bound;
        if (!dist->is_bounded) { lo = -10.0; hi = 10.0; }
        while (uq_dist_cdf(dist, lo) > p) lo -= 5.0;
        while (uq_dist_cdf(dist, hi) < p) hi += 5.0;
        for (int it = 0; it < 100; it++) {
            double mid = 0.5 * (lo + hi);
            double cm = uq_dist_cdf(dist, mid);
            if (cm < p) lo = mid; else hi = mid;
            if (hi - lo < 1e-10) return 0.5 * (lo + hi);
        }
        return 0.5 * (lo + hi);
    }}
}

double uq_dist_log_pdf(UQDistribution* dist, double x) {
    return log(uq_dist_pdf(dist, x) + 1e-300);
}

double uq_dist_sample(UQDistribution* dist) {
    if (!dist) return NAN;
    switch (dist->type) {
    case UQ_DIST_NORMAL:
        return dist->params[0] + dist->params[1] * gauss_rand();
    case UQ_DIST_UNIFORM:
        return dist->params[0] + urand() * (dist->params[1] - dist->params[0]);
    case UQ_DIST_STUDENT_T: {
        /* Ratio of normal to sqrt(chi2/df) */
        double z = gauss_rand();
        double c = 0.0;
        for (int i = 0; i < (int)dist->params[2]; i++)
            c += gauss_rand() * gauss_rand();
        return dist->params[0] + dist->params[1] * z / sqrt(c / dist->params[2]);
    }
    case UQ_DIST_CHI2: {
        double sum = 0.0;
        int n = (int)round(dist->params[0]);
        for (int i = 0; i < n; i++) {
            double z = gauss_rand();
            sum += z * z;
        }
        return sum;
    }
    case UQ_DIST_LOG_NORMAL:
        return exp(dist->params[0] + dist->params[1] * gauss_rand());
    case UQ_DIST_EXPONENTIAL:
        return -log(urand() + 1e-15) / dist->params[0];
    case UQ_DIST_GAMMA: {
        /* Marsaglia-Tsang method for shape > 1 */
        double shape = dist->params[0];
        if (shape < 1.0) {
            /* Use transformation: gamma(a) = gamma(a+1) * U^(1/a) */
            double d = shape + 1.0 - 1.0 / 3.0;
            double c = 1.0 / sqrt(9.0 * d);
            for (;;) {
                double x, v;
                do { x = gauss_rand(); v = 1.0 + c * x; } while (v <= 0.0);
                v = v * v * v;
                double u = urand();
                if (u < 1.0 - 0.0331 * x * x * x * x) return dist->params[1] * d * v * pow(urand(), 1.0 / shape);
                if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) return dist->params[1] * d * v * pow(urand(), 1.0 / shape);
            }
        }
        /* Accept-reject method (Marsaglia-Tsang) */
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / sqrt(9.0 * d);
        for (;;) {
            double x, v;
            do { x = gauss_rand(); v = 1.0 + c * x; } while (v <= 0.0);
            v = v * v * v;
            double u = urand();
            if (u < 1.0 - 0.0331 * x * x * x * x)
                return dist->params[1] * d * v;
            if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
                return dist->params[1] * d * v;
        }
    }
    case UQ_DIST_EMPIRICAL:
        if (!dist->samples || dist->n_samples == 0) return NAN;
        return dist->samples[rand() % dist->n_samples];
    default:
        return uq_dist_quantile(dist, urand());
    }
}

void uq_dist_sample_n(UQDistribution* dist, int n, double* out) {
    for (int i = 0; i < n; i++) out[i] = uq_dist_sample(dist);
}

double uq_dist_entropy_analytical(UQDistribution* dist) {
    if (!dist) return NAN;
    switch (dist->type) {
    case UQ_DIST_NORMAL:
        return 0.5 * log(2.0 * UQ_PI * exp(1.0) * dist->params[1] * dist->params[1]);
    case UQ_DIST_UNIFORM:
        return log(dist->params[1] - dist->params[0]);
    case UQ_DIST_EXPONENTIAL:
        return 1.0 - log(dist->params[0]);
    case UQ_DIST_LOG_NORMAL:
        return dist->params[0] + 0.5 + log(dist->params[1] * sqrt(2.0 * UQ_PI));
    case UQ_DIST_GAMMA:
        return dist->params[0] + log(dist->params[1])
               + uq_stats_lgamma(dist->params[0])
               + (1.0 - dist->params[0]) * uq_stats_digamma(dist->params[0]);
    default:
        return NAN;  /* No analytical form */
    }
}

double uq_dist_kl_divergence(UQDistribution* p, UQDistribution* q, int n_mc) {
    if (!p || !q) return NAN;
    /* Monte Carlo KL estimation: KL(p||q) = E_p[log(p/q)] */
    double sum = 0.0;
    for (int i = 0; i < n_mc; i++) {
        double x = uq_dist_sample(p);
        double lp = uq_dist_log_pdf(p, x);
        double lq = uq_dist_log_pdf(q, x);
        sum += lp - lq;
    }
    return sum / (double)n_mc;
}

UQSummaryStats uq_dist_summary_stats(UQDistribution* dist) {
    UQSummaryStats ss = {0};
    if (!dist) return ss;
    ss.mean = dist->mean;
    ss.variance = dist->variance;
    /* Compute moments via MC for skewed distributions */
    int n_samp = 10000;
    double* samples = (double*)malloc(n_samp * sizeof(double));
    uq_dist_sample_n(dist, n_samp, samples);
    double m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int i = 0; i < n_samp; i++) {
        double d = samples[i] - ss.mean;
        double d2 = d * d;
        m1 += d; m2 += d2; m3 += d2 * d; m4 += d2 * d2;
    }
    m1 /= n_samp; m2 /= n_samp; m3 /= n_samp; m4 /= n_samp;
    ss.skewness = m3 / pow(m2, 1.5);
    ss.kurtosis = m4 / (m2 * m2) - 3.0;
    ss.n_quantiles = 5;
    ss.quantiles = (double*)malloc(5 * sizeof(double));
    double probs[] = {0.025, 0.25, 0.5, 0.75, 0.975};
    for (int i = 0; i < 5; i++)
        ss.quantiles[i] = uq_dist_quantile(dist, probs[i]);
    free(samples);
    return ss;
}

/* ============================================================================
 * Confidence Regions
 * ============================================================================ */

UQConfidenceRegion* uq_ci_create(double confidence, int dimension) {
    UQConfidenceRegion* ci = (UQConfidenceRegion*)calloc(1, sizeof(UQConfidenceRegion));
    ci->confidence_level = confidence;
    ci->dimension = dimension;
    ci->center = (double*)calloc(dimension, sizeof(double));
    if (dimension > 1)
        ci->covariance = uq_matrix_create(dimension, dimension);
    return ci;
}

void uq_ci_free(UQConfidenceRegion* ci) {
    if (!ci) return;
    free(ci->center);
    if (ci->covariance) uq_matrix_free(ci->covariance);
    free(ci);
}

void uq_ci_from_normal(UQConfidenceRegion* ci, double mean, double std, int n) {
    double z = uq_stats_normal_quantile(0.5 + ci->confidence_level * 0.5);
    double se = std / sqrt((double)n);
    ci->lower_bound = mean - z * se;
    ci->upper_bound = mean + z * se;
    ci->center[0] = mean;
    ci->n_samples = n;
}

void uq_ci_from_student(UQConfidenceRegion* ci, double mean, double se, double df) {
    double t = uq_stats_student_t_quantile(0.5 + ci->confidence_level * 0.5, df);
    ci->lower_bound = mean - t * se;
    ci->upper_bound = mean + t * se;
    ci->center[0] = mean;
    ci->dof = df;
}

void uq_ci_from_bootstrap(double* samples, int n, double confidence,
                          UQConfidenceRegion* ci) {
    double* sorted = (double*)malloc(n * sizeof(double));
    memcpy(sorted, samples, n * sizeof(double));
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    double alpha = 1.0 - confidence;
    int lo_idx = (int)(alpha * 0.5 * n);
    int hi_idx = (int)((1.0 - alpha * 0.5) * n - 1);
    if (lo_idx < 0) lo_idx = 0;
    if (hi_idx >= n) hi_idx = n - 1;
    ci->lower_bound = sorted[lo_idx];
    ci->upper_bound = sorted[hi_idx];
    ci->is_asymmetric = true;
    free(sorted);
}

void uq_ci_from_quantiles(double* data, int n, double confidence,
                          UQConfidenceRegion* ci) {
    uq_ci_from_bootstrap(data, n, confidence, ci);
}

bool uq_ci_contains(UQConfidenceRegion* ci, double* point) {
    if (ci->dimension == 1) {
        return point[0] >= ci->lower_bound && point[0] <= ci->upper_bound;
    }
    /* Mahalanobis distance for joint region */
    double md = uq_stats_mahalanobis(point, ci->center, ci->covariance, ci->dimension);
    double crit = uq_stats_chi2_quantile(ci->confidence_level, ci->dimension);
    return md <= crit;
}

double uq_ci_half_width(UQConfidenceRegion* ci) {
    return 0.5 * (ci->upper_bound - ci->lower_bound);
}

void uq_ci_print(UQConfidenceRegion* ci) {
    printf("[%.1f%% CI] ", ci->confidence_level * 100.0);
    if (ci->dimension == 1)
        printf("[%.6f, %.6f] half-width=%.6f\n",
               ci->lower_bound, ci->upper_bound, uq_ci_half_width(ci));
    else
        printf("dim=%d center=[...]\n", ci->dimension);
}

/* ============================================================================
 * Parameter Estimation
 * ============================================================================ */

UQParameterEnsemble* uq_ensemble_create(const char* model_name, int n_params) {
    UQParameterEnsemble* ens = (UQParameterEnsemble*)calloc(1, sizeof(UQParameterEnsemble));
    ens->model_name = strdup(model_name);
    ens->n_params = n_params;
    ens->params = (UQParameterEstimate*)calloc(n_params, sizeof(UQParameterEstimate));
    ens->parameter_covariance = uq_matrix_create(n_params, n_params);
    ens->fisher_information = uq_matrix_create(n_params, n_params);
    return ens;
}

void uq_ensemble_free(UQParameterEnsemble* ens) {
    if (!ens) return;
    free(ens->model_name);
    for (int i = 0; i < ens->n_params; i++) {
        free(ens->params[i].name);
        free(ens->params[i].correlation_row);
    }
    free(ens->params);
    uq_matrix_free(ens->parameter_covariance);
    uq_matrix_free(ens->fisher_information);
    free(ens);
}

void uq_ensemble_set_param(UQParameterEnsemble* ens, int idx, const char* name,
                           double estimate, double std_err) {
    if (idx < 0 || idx >= ens->n_params) return;
    ens->params[idx].name = strdup(name);
    ens->params[idx].nominal_value = estimate;
    ens->params[idx].standard_error = std_err;
    ens->params[idx].relative_error = (fabs(estimate) > 1e-12)
        ? fabs(std_err / estimate) : INFINITY;
}

void uq_ensemble_compute_statistics(UQParameterEnsemble* ens) {
    for (int i = 0; i < ens->n_params; i++) {
        double est = ens->params[i].nominal_value;
        double se = ens->params[i].standard_error;
        if (fabs(se) < 1e-12) {
            ens->params[i].t_statistic = INFINITY;
            ens->params[i].p_value = 0.0;
            continue;
        }
        double t = est / se;
        ens->params[i].t_statistic = t;
        /* Two-sided p-value from t-distribution */
        double pt = uq_dist_cdf(
            uq_dist_create_student_t(0.0, 1.0, ens->n_observations - ens->n_params),
            fabs(t));
        ens->params[i].p_value = 2.0 * (1.0 - pt);
    }
}

void uq_ensemble_compute_correlations(UQParameterEnsemble* ens) {
    for (int i = 0; i < ens->n_params; i++) {
        ens->params[i].n_correlated = ens->n_params - 1;
        ens->params[i].correlation_row = (double*)malloc(
            (ens->n_params - 1) * sizeof(double));
        for (int j = 0, k = 0; j < ens->n_params; j++) {
            if (j == i) continue;
            double cov_ij = uq_matrix_get(ens->parameter_covariance, i, j);
            double sd_i = sqrt(uq_matrix_get(ens->parameter_covariance, i, i));
            double sd_j = sqrt(uq_matrix_get(ens->parameter_covariance, j, j));
            ens->params[i].correlation_row[k++] = (sd_i > 1e-15 && sd_j > 1e-15)
                ? cov_ij / (sd_i * sd_j) : 0.0;
        }
    }
}

void uq_ensemble_print(UQParameterEnsemble* ens) {
    printf("Parameter Ensemble: %s (%d params, n=%d)\n",
           ens->model_name, ens->n_params, ens->n_observations);
    printf("  R²=%.4f  R²_adj=%.4f  AIC=%.2f  BIC=%.2f\n",
           ens->r_squared, ens->adjusted_r_squared, ens->akaike_ic, ens->bayesian_ic);
    printf("  %-20s %12s %12s %10s %10s\n", "Parameter", "Estimate", "Std.Err", "t-value", "p-value");
    for (int i = 0; i < ens->n_params; i++)
        printf("  %-20s %12.6f %12.6f %10.3f %10.4f\n",
               ens->params[i].name, ens->params[i].nominal_value,
               ens->params[i].standard_error, ens->params[i].t_statistic,
               ens->params[i].p_value);
}

double uq_ensemble_mahalanobis_distance(UQParameterEnsemble* ens, double* point) {
    double* diff = (double*)malloc(ens->n_params * sizeof(double));
    for (int i = 0; i < ens->n_params; i++)
        diff[i] = point[i] - ens->params[i].nominal_value;
    double md = uq_stats_mahalanobis(diff, NULL, ens->parameter_covariance,
                                     ens->n_params);
    free(diff);
    return md;
}

/* ============================================================================
 * Linear Model UQ
 * ============================================================================ */

UQLinearModel* uq_lm_create(UQMatrix* X, UQVector* y) {
    UQLinearModel* lm = (UQLinearModel*)calloc(1, sizeof(UQLinearModel));
    lm->n = X->rows;
    lm->p = X->cols;
    lm->design_matrix = uq_matrix_copy(X);
    lm->response = (UQVector*)malloc(sizeof(UQVector));
    lm->response->dimension = y->dimension;
    lm->response->components = (double*)malloc(y->dimension * sizeof(double));
    memcpy(lm->response->components, y->components, y->dimension * sizeof(double));
    lm->coefficients = uq_vector_create(lm->p);
    lm->covariance_beta = uq_matrix_create(lm->p, lm->p);
    lm->residuals = (double*)calloc(lm->n, sizeof(double));
    lm->leverages = (double*)calloc(lm->n, sizeof(double));
    lm->studentized_residuals = (double*)calloc(lm->n, sizeof(double));
    lm->cook_distance = (double*)calloc(lm->n, sizeof(double));
    lm->variance_inflation_factors = (double*)calloc(lm->p, sizeof(double));
    return lm;
}

void uq_lm_free(UQLinearModel* lm) {
    if (!lm) return;
    uq_matrix_free(lm->design_matrix);
    uq_vector_free(lm->response);
    uq_vector_free(lm->coefficients);
    uq_matrix_free(lm->covariance_beta);
    free(lm->residuals);
    free(lm->leverages);
    free(lm->studentized_residuals);
    free(lm->cook_distance);
    free(lm->variance_inflation_factors);
    free(lm);
}

void uq_lm_fit(UQLinearModel* lm) {
    /* OLS: β̂ = (XᵀX)⁻¹ Xᵀy */
    int n = lm->n, p = lm->p;
    UQMatrix* X = lm->design_matrix;
    UQVector* y = lm->response;

    /* XᵀX */
    UQMatrix* XtX = uq_matrix_create(p, p);
    UQMatrix* Xt = uq_matrix_create(p, n);
    uq_matrix_transpose(Xt, X);
    uq_matrix_multiply(XtX, Xt, X);

    /* Rank */
    lm->rank = uq_matrix_rank(XtX, 1e-10);

    /* (XᵀX)⁻¹ */
    UQMatrix* XtX_inv = uq_matrix_create(p, p);
    uq_matrix_invert(XtX_inv, XtX);

    /* Xᵀy */
    UQVector* Xty = uq_vector_create(p);
    for (int i = 0; i < p; i++) {
        double s = 0.0;
        for (int k = 0; k < n; k++)
            s += uq_matrix_get(Xt, i, k) * y->components[k];
        Xty->components[i] = s;
    }

    /* β̂ = (XᵀX)⁻¹ Xᵀy */
    for (int i = 0; i < p; i++) {
        double s = 0.0;
        for (int j = 0; j < p; j++)
            s += uq_matrix_get(XtX_inv, i, j) * Xty->components[j];
        lm->coefficients->components[i] = s;
    }

    /* Residuals */
    double rss = 0.0;
    for (int i = 0; i < n; i++) {
        double yh = 0.0;
        for (int j = 0; j < p; j++)
            yh += uq_matrix_get(X, i, j) * lm->coefficients->components[j];
        lm->residuals[i] = y->components[i] - yh;
        rss += lm->residuals[i] * lm->residuals[i];
    }
    lm->sigma_squared = rss / (double)(n - p);

    /* Cov(β̂) = σ² (XᵀX)⁻¹ */
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++)
            uq_matrix_set(lm->covariance_beta, i, j,
                lm->sigma_squared * uq_matrix_get(XtX_inv, i, j));

    /* Leverages (hat matrix diagonal) */
    for (int i = 0; i < n; i++) {
        double h_ii = 0.0;
        for (int j = 0; j < p; j++)
            for (int k = 0; k < p; k++)
                h_ii += uq_matrix_get(X, i, j) * uq_matrix_get(XtX_inv, j, k)
                        * uq_matrix_get(X, i, k);
        lm->leverages[i] = h_ii;
    }

    /* Studentized residuals */
    for (int i = 0; i < n; i++)
        lm->studentized_residuals[i] = lm->residuals[i]
            / sqrt(lm->sigma_squared * (1.0 - lm->leverages[i]));

    uq_matrix_free(XtX);
    uq_matrix_free(XtX_inv);
    uq_matrix_free(Xt);
    uq_vector_free(Xty);
}

void uq_lm_fit_weighted(UQLinearModel* lm, double* weights) {
    /* WLS: β̂ = (XᵀWX)⁻¹ XᵀWy */
    int n = lm->n, p = lm->p;
    UQMatrix* X = lm->design_matrix;
    UQVector* y = lm->response;

    UQMatrix* XtW = uq_matrix_create(p, n);
    for (int i = 0; i < p; i++)
        for (int k = 0; k < n; k++)
            uq_matrix_set(XtW, i, k, uq_matrix_get(X, k, i) * weights[k]);

    UQMatrix* XtWX = uq_matrix_create(p, p);
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += uq_matrix_get(XtW, i, k) * uq_matrix_get(X, k, j);
            uq_matrix_set(XtWX, i, j, s);
        }

    UQMatrix* XtWX_inv = uq_matrix_create(p, p);
    uq_matrix_invert(XtWX_inv, XtWX);

    /* β̂ */
    for (int i = 0; i < p; i++) {
        double s = 0.0;
        for (int k = 0; k < n; k++)
            s += uq_matrix_get(XtW, i, k) * y->components[k];
        lm->coefficients->components[i] = 0.0;
        for (int j = 0; j < p; j++)
            lm->coefficients->components[i] += uq_matrix_get(XtWX_inv, i, j) * s;
    }

    uq_matrix_free(XtW);
    uq_matrix_free(XtWX);
    uq_matrix_free(XtWX_inv);
}

void uq_lm_predict(UQLinearModel* lm, double* x_new,
                   double* y_hat, double* se_fit, double* se_pred) {
    if (!lm || !x_new) return;
    double yh = 0.0;
    for (int j = 0; j < lm->p; j++)
        yh += x_new[j] * lm->coefficients->components[j];
    if (y_hat) *y_hat = yh;

    /* Variance of fitted value: σ² * x_newᵀ (XᵀX)⁻¹ x_new */
    double var_fit = 0.0;
    for (int i = 0; i < lm->p; i++)
        for (int j = 0; j < lm->p; j++)
            var_fit += x_new[i] * uq_matrix_get(lm->covariance_beta, i, j) * x_new[j];
    /* cov_beta already has σ² factor */
    if (se_fit) *se_fit = sqrt(var_fit);
    /* Variance of prediction: same + σ² */
    double var_pred = var_fit + lm->sigma_squared;
    if (se_pred) *se_pred = sqrt(var_pred);
}

void uq_lm_anova(UQLinearModel* lm, double* ss_reg, double* ss_res,
                 double* ss_tot, double* f_stat, double* p_val) {
    double ybar = 0.0;
    for (int i = 0; i < lm->n; i++)
        ybar += lm->response->components[i];
    ybar /= lm->n;

    double ssr = 0.0, sse = 0.0, sst = 0.0;
    for (int i = 0; i < lm->n; i++) {
        double yh = lm->response->components[i] - lm->residuals[i];
        ssr += (yh - ybar) * (yh - ybar);
        sse += lm->residuals[i] * lm->residuals[i];
        sst += (lm->response->components[i] - ybar) * (lm->response->components[i] - ybar);
    }
    if (ss_reg) *ss_reg = ssr;
    if (ss_res) *ss_res = sse;
    if (ss_tot) *ss_tot = sst;

    if (f_stat) {
        double msr = ssr / (lm->p - 1.0);
        double mse = sse / (double)(lm->n - lm->p);
        *f_stat = (mse > 1e-15) ? msr / mse : INFINITY;
    }
    if (p_val) *p_val = 0.0;  /* Placeholder — F-dist quantile needed */
}

void uq_lm_influence_diagnostics(UQLinearModel* lm) {
    for (int i = 0; i < lm->n; i++) {
        double r = lm->studentized_residuals[i];
        double h = lm->leverages[i];
        double p = (double)lm->p;
        lm->cook_distance[i] = r * r * h / (p * (1.0 - h) * (1.0 - h));
    }
}

void uq_lm_vif(UQLinearModel* lm) {
    /* VIF_j = 1 / (1 - R²_j) where R²_j = R² of X_j ~ X_{-j} */
    UQMatrix* X = lm->design_matrix;
    int n = lm->n, p = lm->p;
    for (int j = 0; j < p; j++) {
        /* Build X_{-j}: all columns except j */
        UQMatrix* X_minus = uq_matrix_create(n, p - 1);
        for (int r = 0; r < n; r++) {
            int c2 = 0;
            for (int c = 0; c < p; c++) {
                if (c == j) continue;
                uq_matrix_set(X_minus, r, c2++, uq_matrix_get(X, r, c));
            }
        }
        UQVector* xj = uq_vector_create(n);
        for (int r = 0; r < n; r++)
            xj->components[r] = uq_matrix_get(X, r, j);

        /* μ = mean(x_j) */
        double mu = 0.0;
        for (int r = 0; r < n; r++) mu += xj->components[r];
        mu /= n;

        /* Fit x_j ~ X_{-j} */
        UQLinearModel* aux_lm = uq_lm_create(X_minus, xj);
        uq_lm_fit(aux_lm);

        /* R² */
        double sse = 0.0, sst = 0.0;
        for (int r = 0; r < n; r++) {
            sse += aux_lm->residuals[r] * aux_lm->residuals[r];
            double yd = xj->components[r] - mu;
            sst += yd * yd;
        }
        double r2 = 1.0 - sse / (sst + 1e-15);
        lm->variance_inflation_factors[j] = (r2 < 0.99999)
            ? 1.0 / (1.0 - r2) : INFINITY;

        uq_matrix_free(X_minus);
        uq_vector_free(xj);
        uq_lm_free(aux_lm);
    }
}

double uq_lm_press(UQLinearModel* lm) {
    double press = 0.0;
    for (int i = 0; i < lm->n; i++) {
        double r = lm->residuals[i];
        double h = lm->leverages[i];
        double r_loo = r / (1.0 - h + 1e-15);
        press += r_loo * r_loo;
    }
    return press;
}

/* ============================================================================
 * Convergence Diagnostics
 * ============================================================================ */

UQConvergenceDiagnostic* uq_conv_create(int n_lags) {
    UQConvergenceDiagnostic* d = (UQConvergenceDiagnostic*)calloc(1, sizeof(UQConvergenceDiagnostic));
    d->n_lags = n_lags;
    d->autocorrelation = (double*)calloc(n_lags, sizeof(double));
    return d;
}

void uq_conv_free(UQConvergenceDiagnostic* diag) {
    if (!diag) return;
    free(diag->autocorrelation);
    free(diag);
}

void uq_conv_geweke(UQConvergenceDiagnostic* diag, double* chain, int n,
                    double frac_first, double frac_last) {
    int n1 = (int)(frac_first * n);
    int n2 = (int)(frac_last * n);
    double m1 = 0.0, m2 = 0.0, v1 = 0.0, v2 = 0.0;
    for (int i = 0; i < n1; i++) m1 += chain[i];
    for (int i = n - n2; i < n; i++) m2 += chain[i];
    m1 /= n1; m2 /= n2;
    for (int i = 0; i < n1; i++) v1 += (chain[i] - m1) * (chain[i] - m1);
    for (int i = n - n2; i < n; i++) v2 += (chain[i] - m2) * (chain[i] - m2);
    v1 /= n1; v2 /= n2;
    /* Spectral density at 0 — simplified: use var estimates with batch means */
    double se = sqrt(v1 / n1 + v2 / n2);
    diag->geweke_z = (fabs(se) > 1e-15) ? (m1 - m2) / se : 0.0;
    diag->converged = fabs(diag->geweke_z) < 1.96;
}

void uq_conv_gelman_rubin(UQConvergenceDiagnostic* diag, double** chains,
                          int n_chains, int n_per_chain) {
    /* Between-chain variance B/n */
    double* chain_means = (double*)malloc(n_chains * sizeof(double));
    double overall_mean = 0.0;
    for (int c = 0; c < n_chains; c++) {
        chain_means[c] = 0.0;
        for (int i = 0; i < n_per_chain; i++)
            chain_means[c] += chains[c][i];
        chain_means[c] /= n_per_chain;
        overall_mean += chain_means[c];
    }
    overall_mean /= n_chains;

    double B = 0.0;
    for (int c = 0; c < n_chains; c++)
        B += (chain_means[c] - overall_mean) * (chain_means[c] - overall_mean);
    B *= (double)n_per_chain / (double)(n_chains - 1);

    /* Within-chain variance W */
    double W = 0.0;
    for (int c = 0; c < n_chains; c++) {
        double cv = 0.0;
        for (int i = 0; i < n_per_chain; i++)
            cv += (chains[c][i] - chain_means[c]) * (chains[c][i] - chain_means[c]);
        W += cv / (double)(n_per_chain - 1);
    }
    W /= n_chains;

    double var_plus = ((double)(n_per_chain - 1) / (double)n_per_chain) * W
                      + B / (double)n_per_chain;
    diag->gelman_rubin_rhat = (W > 1e-15) ? sqrt(var_plus / W) : 1.0;
    diag->converged = diag->gelman_rubin_rhat < 1.1;
    free(chain_means);
}

void uq_conv_autocorrelation(UQConvergenceDiagnostic* diag, double* chain, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) m += chain[i];
    m /= n;

    double v0 = 0.0;
    for (int i = 0; i < n; i++) v0 += (chain[i] - m) * (chain[i] - m);
    v0 /= n;

    int max_lag = (diag->n_lags < n / 4) ? diag->n_lags : n / 4;
    for (int lag = 0; lag < max_lag; lag++) {
        double cv = 0.0;
        for (int i = 0; i < n - lag; i++)
            cv += (chain[i] - m) * (chain[i + lag] - m);
        diag->autocorrelation[lag] = (v0 > 1e-15) ? cv / (n * v0) : 0.0;
    }
}

int uq_conv_effective_sample_size(UQConvergenceDiagnostic* diag, double* chain, int n) {
    uq_conv_autocorrelation(diag, chain, n);
    double sum_rho = 0.0;
    for (int lag = 1; lag < diag->n_lags; lag++) {
        if (diag->autocorrelation[lag] < 0.01) break;
        sum_rho += diag->autocorrelation[lag];
    }
    double ess = (double)n / (1.0 + 2.0 * sum_rho);
    diag->effective_sample_size = (int)ess;
    return diag->effective_sample_size;
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

UQMatrix* uq_matrix_create(int rows, int cols) {
    UQMatrix* m = (UQMatrix*)calloc(1, sizeof(UQMatrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));
    m->is_owner = true;
    return m;
}

void uq_matrix_free(UQMatrix* mat) {
    if (!mat) return;
    if (mat->is_owner) free(mat->data);
    free(mat);
}

UQMatrix* uq_matrix_copy(UQMatrix* src) {
    UQMatrix* dst = uq_matrix_create(src->rows, src->cols);
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(double));
    return dst;
}

void uq_matrix_set(UQMatrix* mat, int i, int j, double val) {
    if (i >= 0 && i < mat->rows && j >= 0 && j < mat->cols)
        mat->data[i * mat->cols + j] = val;
}

double uq_matrix_get(UQMatrix* mat, int i, int j) {
    if (i >= 0 && i < mat->rows && j >= 0 && j < mat->cols)
        return mat->data[i * mat->cols + j];
    return NAN;
}

void uq_matrix_multiply(UQMatrix* C, UQMatrix* A, UQMatrix* B) {
    /* C = A * B,  A: m×n, B: n×p → C: m×p */
    int m = A->rows, n = A->cols, p = B->cols;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += uq_matrix_get(A, i, k) * uq_matrix_get(B, k, j);
            uq_matrix_set(C, i, j, s);
        }
}

void uq_matrix_transpose(UQMatrix* At, UQMatrix* A) {
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            uq_matrix_set(At, j, i, uq_matrix_get(A, i, j));
}

void uq_matrix_invert(UQMatrix* A_inv, UQMatrix* A) {
    int n = A->rows;
    /* Gauss-Jordan with partial pivoting */
    double* aug = (double*)calloc(n * n * 2, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (2 * n) + j] = uq_matrix_get(A, i, j);
            aug[i * (2 * n) + n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < n; col++) {
        /* Pivot */
        int max_row = col;
        double max_val = fabs(aug[col * (2 * n) + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(aug[row * (2 * n) + col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-14) continue;
        if (max_row != col) {
            for (int j = 0; j < 2 * n; j++) {
                double t = aug[col * (2 * n) + j];
                aug[col * (2 * n) + j] = aug[max_row * (2 * n) + j];
                aug[max_row * (2 * n) + j] = t;
            }
        }
        double piv = aug[col * (2 * n) + col];
        for (int j = 0; j < 2 * n; j++)
            aug[col * (2 * n) + j] /= piv;
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double f = aug[row * (2 * n) + col];
            for (int j = 0; j < 2 * n; j++)
                aug[row * (2 * n) + j] -= f * aug[col * (2 * n) + j];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            uq_matrix_set(A_inv, i, j, aug[i * (2 * n) + n + j]);
    free(aug);
}

double uq_matrix_determinant(UQMatrix* A) {
    int n = A->rows;
    if (n == 1) return uq_matrix_get(A, 0, 0);
    if (n == 2)
        return uq_matrix_get(A,0,0) * uq_matrix_get(A,1,1)
               - uq_matrix_get(A,0,1) * uq_matrix_get(A,1,0);
    /* LU with partial pivoting */
    double* LU = (double*)malloc(n * n * sizeof(double));
    memcpy(LU, A->data, n * n * sizeof(double));
    double det = 1.0;
    int sign = 1;
    for (int k = 0; k < n; k++) {
        int max_i = k;
        double max_v = fabs(LU[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = fabs(LU[i * n + k]);
            if (v > max_v) { max_v = v; max_i = i; }
        }
        if (max_v < 1e-15) { free(LU); return 0.0; }
        if (max_i != k) {
            sign = -sign;
            for (int j = 0; j < n; j++) {
                double t = LU[k * n + j];
                LU[k * n + j] = LU[max_i * n + j];
                LU[max_i * n + j] = t;
            }
        }
        for (int i = k + 1; i < n; i++) {
            LU[i * n + k] /= LU[k * n + k];
            for (int j = k + 1; j < n; j++)
                LU[i * n + j] -= LU[i * n + k] * LU[k * n + j];
        }
    }
    for (int i = 0; i < n; i++) det *= LU[i * n + i];
    free(LU);
    return det * sign;
}

double uq_matrix_trace(UQMatrix* A) {
    double tr = 0.0;
    int n = (A->rows < A->cols) ? A->rows : A->cols;
    for (int i = 0; i < n; i++)
        tr += uq_matrix_get(A, i, i);
    return tr;
}

void uq_matrix_cholesky(UQMatrix* L, UQMatrix* A) {
    int n = A->rows;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += uq_matrix_get(L, i, k) * uq_matrix_get(L, j, k);
            if (i == j)
                uq_matrix_set(L, i, j, sqrt(uq_matrix_get(A, i, i) - sum + 1e-14));
            else
                uq_matrix_set(L, i, j, (uq_matrix_get(A, i, j) - sum)
                    / (uq_matrix_get(L, j, j) + 1e-14));
        }
    }
}

void uq_matrix_eigen_sym(UQMatrix* A, double* eigenvalues, UQMatrix* eigenvectors) {
    /* Jacobi iteration for symmetric matrices */
    int n = A->rows;
    double* V = (double*)calloc(n * n, sizeof(double));
    double* a = (double*)malloc(n * n * sizeof(double));
    memcpy(a, A->data, n * n * sizeof(double));
    for (int i = 0; i < n; i++) V[i * n + i] = 1.0;

    for (int sweep = 0; sweep < 50; sweep++) {
        double max_off = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double v = fabs(a[i * n + j]);
                if (v > max_off) { max_off = v; p = i; q = j; }
            }
        if (max_off < 1e-12) break;

        double phi = 0.5 * atan2(2.0 * a[p * n + q], a[p * n + p] - a[q * n + q]);
        double c = cos(phi), s = sin(phi);

        /* Rotate A */
        double app = a[p * n + p], aqq = a[q * n + q], apq = a[p * n + q];
        a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = a[q * n + p] = 0.0;

        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double aip = a[i * n + p], aiq = a[i * n + q];
                a[i * n + p] = a[p * n + i] = c * aip - s * aiq;
                a[i * n + q] = a[q * n + i] = s * aip + c * aiq;
            }
        }
        /* Accumulate eigenvectors */
        for (int i = 0; i < n; i++) {
            double vip = V[i * n + p], viq = V[i * n + q];
            V[i * n + p] = c * vip - s * viq;
            V[i * n + q] = s * vip + c * viq;
        }
    }
    for (int i = 0; i < n; i++) eigenvalues[i] = a[i * n + i];
    if (eigenvectors)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                uq_matrix_set(eigenvectors, i, j, V[i * n + j]);
    free(V);
    free(a);
}

void uq_matrix_svd(UQMatrix* U, double* S, UQMatrix* Vt, UQMatrix* A) {
    /* Simplified: only up to 4x4 via characteristic eqn */
    int m = A->rows, n = A->cols;
    UQMatrix* AtA = uq_matrix_create(n, n);
    UQMatrix* At = uq_matrix_create(n, m);
    uq_matrix_transpose(At, A);
    uq_matrix_multiply(AtA, At, A);

    double* eigenvalues = (double*)malloc(n * sizeof(double));
    uq_matrix_eigen_sym(AtA, eigenvalues, Vt);
    for (int j = 0; j < n; j++) {
        S[j] = (eigenvalues[j] > 1e-14) ? sqrt(eigenvalues[j]) : 0.0;
    }

    /* U = A * V * diag(1/S) */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += uq_matrix_get(A, i, k) * uq_matrix_get(Vt, j, k);
            uq_matrix_set(U, i, j, (S[j] > 1e-14) ? s / S[j] : 0.0);
        }

    uq_matrix_free(AtA);
    uq_matrix_free(At);
    free(eigenvalues);
}

double uq_matrix_condition_number(UQMatrix* A) {
    int n = (A->rows < A->cols) ? A->rows : A->cols;
    double* S = (double*)malloc(n * sizeof(double));
    UQMatrix* U = uq_matrix_create(A->rows, n);
    UQMatrix* Vt = uq_matrix_create(n, A->cols);
    uq_matrix_svd(U, S, Vt, A);
    double cn = (S[n-1] > 1e-15) ? S[0] / S[n-1] : INFINITY;
    uq_matrix_free(U);
    uq_matrix_free(Vt);
    free(S);
    return cn;
}

int uq_matrix_rank(UQMatrix* A, double tol) {
    int n = (A->rows < A->cols) ? A->rows : A->cols;
    double* S = (double*)malloc(n * sizeof(double));
    UQMatrix* U = uq_matrix_create(A->rows, n);
    UQMatrix* Vt = uq_matrix_create(n, A->cols);
    uq_matrix_svd(U, S, Vt, A);
    int rank = 0;
    double smax = S[0];
    for (int i = 0; i < n; i++)
        if (S[i] > tol * smax) rank++;
    uq_matrix_free(U);
    uq_matrix_free(Vt);
    free(S);
    return rank;
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

UQVector* uq_vector_create(int dim) {
    UQVector* v = (UQVector*)calloc(1, sizeof(UQVector));
    v->dimension = dim;
    v->components = (double*)calloc(dim, sizeof(double));
    return v;
}

void uq_vector_free(UQVector* v) {
    if (!v) return;
    free(v->components);
    free(v);
}

double uq_vector_norm(UQVector* v) {
    double s = 0.0;
    for (int i = 0; i < v->dimension; i++)
        s += v->components[i] * v->components[i];
    return sqrt(s);
}

double uq_vector_dot(UQVector* a, UQVector* b) {
    double s = 0.0;
    int n = (a->dimension < b->dimension) ? a->dimension : b->dimension;
    for (int i = 0; i < n; i++)
        s += a->components[i] * b->components[i];
    return s;
}

void uq_vector_scale(UQVector* v, double s) {
    for (int i = 0; i < v->dimension; i++)
        v->components[i] *= s;
}

/* ============================================================================
 * Statistical Functions
 * ============================================================================ */

double uq_stats_lgamma(double x) {
    /* Stirling series for log(Gamma(x)), x > 0 */
    if (x <= 0.0) return NAN;
    static const double c[] = {
        0.0, 1.0 / 12.0, -1.0 / 360.0, 1.0 / 1260.0, -1.0 / 1680.0,
        1.0 / 1188.0, -691.0 / 360360.0, 1.0 / 156.0
    };
    double s = 0.5 * log(2.0 * UQ_PI) - x + (x - 0.5) * log(x);
    double t = 0.0, v = 1.0 / (x * x);
    for (int i = 1; i <= 6; i++) {
        t = t * v + c[i];
    }
    return s + t / x;
}

double uq_stats_digamma(double x) {
    /* Asymptotic expansion */
    if (x <= 0.0) return NAN;
    double inv_x2 = 1.0 / (x * x);
    return log(x) - 0.5 / x - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0
           - inv_x2 * (1.0 / 252.0)));
}

double uq_stats_erfinv(double x) {
    /* Approximation from Giles (2016) */
    double w = -log((1.0 - x) * (1.0 + x));
    double p;
    if (w < 5.0) {
        w -= 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    } else {
        w = sqrt(w) - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    return p * x;
}

double uq_stats_normal_quantile(double p) {
    if (p <= 0.0) return -10.0;
    if (p >= 1.0) return 10.0;
    /* Moro (1995) approximation */
    double y = p - 0.5;
    double r;
    if (fabs(y) < 0.42) {
        r = y * y;
        double a[] = {2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637};
        double b[] = {-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833};
        double num = a[0] + r * (a[1] + r * (a[2] + r * a[3]));
        double den = 1.0 + r * (b[0] + r * (b[1] + r * (b[2] + r * b[3])));
        return y * num / den;
    } else {
        r = (y < 0.0) ? p : 1.0 - p;
        r = sqrt(-log(r));
        double c[] = {1.432788187, 0.189269265, 0.00130811};
        double d[] = {3.654152846, 2.655698504, 0.268419342};
        double num = c[0] + r * (c[1] + r * c[2]);
        double den = 1.0 + r * (d[0] + r * (d[1] + r * d[2]));
        return (y < 0.0) ? -(num / den) : num / den;
    }
}

double uq_stats_student_t_quantile(double p, double df) {
    if (df <= 0.0) return NAN;
    double z = uq_stats_normal_quantile(p);
    /* Cornish-Fisher expansion */
    double z2 = z * z, z3 = z2 * z, z5 = z3 * z2;
    return z + (z3 + z) / (4.0 * df) + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df * df)
           + (3.0 * z5 * z2 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df * df * df);
}

double uq_stats_chi2_quantile(double p, double df) {
    /* Wilson-Hilferty approximation */
    double z = uq_stats_normal_quantile(p);
    double m = 1.0 - 2.0 / (9.0 * df);
    return df * m * m * m + z * sqrt(2.0 * df / (9.0 * df)) * m * m;
}

double uq_stats_f_quantile(double p, double df1, double df2) {
    double x1 = uq_stats_chi2_quantile(p, df1);
    double x2 = uq_stats_chi2_quantile(p, df2);
    return (x1 / df1) / (x2 / df2);
}

double uq_stats_beta_regularized(double a, double b, double x) {
    /* Continued fraction representation */
    if (x < 0.0 || x > 1.0) return NAN;
    if (x == 0.0 || x == 1.0) return x;

    double log_beta = uq_stats_lgamma(a) + uq_stats_lgamma(b) - uq_stats_lgamma(a + b);
    double front = exp(log(a) + (a - 1.0) * log(x) + (b - 1.0) * log(1.0 - x) - log_beta);

    /* Lentz continued fraction */
    double f = 1.0, C = 1.0, D = 1.0;
    for (int m = 0; m < 200; m++) {
        double d_term;
        if (m == 0) {
            d_term = 1.0;
        } else if (m % 2 == 0) {
            int mm = m / 2;
            d_term = (mm * (b - mm) * x) / ((a + 2.0 * mm - 1.0) * (a + 2.0 * mm));
        } else {
            int mm = (m + 1) / 2;
            d_term = -((a + mm - 1.0) * (a + b + mm - 1.0) * x)
                     / ((a + 2.0 * mm - 2.0) * (a + 2.0 * mm - 1.0));
        }
        D = 1.0 + d_term * D;
        if (fabs(D) < 1e-30) D = 1e-30;
        C = 1.0 + d_term / C;
        if (fabs(C) < 1e-30) C = 1e-30;
        D = 1.0 / D;
        double delta = C * D;
        f *= delta;
        if (fabs(delta - 1.0) < 1e-14) break;
    }
    return front * (f - 1.0) / a;
}

double uq_stats_correlation(double* x, double* y, int n) {
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - mx, dy = y[i] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    return sxy / sqrt(sxx * syy + 1e-15);
}

void uq_stats_covariance_matrix(UQMatrix* cov, UQMatrix* data) {
    /* data: n_obs × n_vars, cov: n_vars × n_vars */
    int n = data->rows, p = data->cols;
    double* means = (double*)calloc(p, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            means[j] += uq_matrix_get(data, i, j);
    for (int j = 0; j < p; j++) means[j] /= n;

    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += (uq_matrix_get(data, k, i) - means[i])
                     * (uq_matrix_get(data, k, j) - means[j]);
            uq_matrix_set(cov, i, j, s / (double)(n - 1));
        }
    free(means);
}

double uq_stats_mahalanobis(double* x, double* mu, UQMatrix* sigma, int dim) {
    double* diff = (double*)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++)
        diff[i] = x[i] - (mu ? mu[i] : 0.0);

    UQMatrix* sigma_inv = uq_matrix_create(dim, dim);
    uq_matrix_invert(sigma_inv, sigma);

    double md2 = 0.0;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            md2 += diff[i] * uq_matrix_get(sigma_inv, i, j) * diff[j];

    uq_matrix_free(sigma_inv);
    free(diff);
    return sqrt(md2 + 1e-15);
}

/* ============================================================================
 * Data Operations
 * ============================================================================ */

UQDataset* uq_dataset_create(int n, int input_dim) {
    UQDataset* ds = (UQDataset*)calloc(1, sizeof(UQDataset));
    ds->n_points = n;
    ds->input_dimension = input_dim;
    ds->x = (double*)calloc(n * input_dim, sizeof(double));
    ds->y = (double*)calloc(n, sizeof(double));
    ds->y_std = (double*)calloc(n, sizeof(double));
    ds->input_names = (char**)calloc(input_dim, sizeof(char*));
    return ds;
}

void uq_dataset_free(UQDataset* ds) {
    if (!ds) return;
    free(ds->x);
    free(ds->y);
    free(ds->y_std);
    for (int i = 0; i < ds->input_dimension; i++)
        free(ds->input_names[i]);
    free(ds->input_names);
    free(ds);
}

void uq_dataset_split(UQDataset* ds, double train_frac,
                      UQDataset** train, UQDataset** test) {
    int n_train = (int)(train_frac * ds->n_points);
    int n_test = ds->n_points - n_train;
    int dim = ds->input_dimension;

    *train = uq_dataset_create(n_train, dim);
    *test = uq_dataset_create(n_test, dim);

    /* Simple split: first n_train to train, rest to test */
    memcpy((*train)->x, ds->x, n_train * dim * sizeof(double));
    memcpy((*train)->y, ds->y, n_train * sizeof(double));
    memcpy((*test)->x, ds->x + n_train * dim, n_test * dim * sizeof(double));
    memcpy((*test)->y, ds->y + n_train, n_test * sizeof(double));
}

void uq_dataset_standardize(UQDataset* ds) {
    int n = ds->n_points, d = ds->input_dimension;
    for (int j = 0; j < d; j++) {
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < n; i++)
            mean += ds->x[i * d + j];
        mean /= n;
        for (int i = 0; i < n; i++)
            var += (ds->x[i * d + j] - mean) * (ds->x[i * d + j] - mean);
        var /= n;
        double std = sqrt(var);
        if (std > 1e-15)
            for (int i = 0; i < n; i++)
                ds->x[i * d + j] = (ds->x[i * d + j] - mean) / std;
    }
}

void uq_dataset_summary(UQDataset* ds, UQSummaryStats** x_stats,
                         UQSummaryStats* y_stats) {
    int n = ds->n_points, d = ds->input_dimension;
    if (x_stats) {
        *x_stats = (UQSummaryStats*)malloc(d * sizeof(UQSummaryStats));
        for (int j = 0; j < d; j++) {
            /* Column-stride access for each variable */
            double* col = (double*)malloc(n * sizeof(double));
            for (int i = 0; i < n; i++) col[i] = ds->x[i * d + j];
            UQDistribution* td = uq_dist_create_empirical(col, n);
            (*x_stats)[j] = uq_dist_summary_stats(td);
            uq_dist_free(td);
            free(col);
        }
    }
    if (y_stats) {
        UQDistribution* td = uq_dist_create_empirical(ds->y, n);
        *y_stats = uq_dist_summary_stats(td);
        uq_dist_free(td);
    }
}
