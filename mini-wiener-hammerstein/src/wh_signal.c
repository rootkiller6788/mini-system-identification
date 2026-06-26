/**
 * wh_signal.c ? Signal Design and Generation
 *
 * Implements excitation signal generation for system identification:
 * multisines (standard, odd, full), chirps, PRBS, colored noise,
 * Gaussian noise, and conditioning functions.
 *
 * Knowledge Level: L3 (Mathematical Structures), L5 (Algorithms)
 */

#include "wh_signal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ??? Internal PRNG state ???????????????????????????????????????????????? */

static unsigned int g_sig_rand = 12345;

static void sig_srand(unsigned int seed) {
    if (seed == 0) seed = (unsigned int)time(NULL);
    g_sig_rand = seed;
}

static double sig_rand(void) {
    g_sig_rand = 1103515245 * g_sig_rand + 12345;
    return (double)(g_sig_rand & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

static double sig_randn(void) {
    double u1 = sig_rand();
    double u2 = sig_rand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ??? Signal statistics ?????????????????????????????????????????????????? */

double wh_signal_mean(const double* x, int N) {
    if (!x || N <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += x[i];
    return sum / N;
}

double wh_signal_variance(const double* x, int N) {
    if (!x || N <= 1) return 0.0;
    double mean = wh_signal_mean(x, N);
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        double d = x[i] - mean;
        sum += d * d;
    }
    return sum / (N - 1);
}

double wh_signal_rms(const double* x, int N) {
    if (!x || N <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += x[i] * x[i];
    return sqrt(sum / N);
}

double wh_signal_peak(const double* x, int N) {
    if (!x || N <= 0) return 0.0;
    double peak = fabs(x[0]);
    for (int i = 1; i < N; i++) {
        if (fabs(x[i]) > peak) peak = fabs(x[i]);
    }
    return peak;
}

double wh_signal_crest_factor(const double* x, int N) {
    double rms = wh_signal_rms(x, N);
    if (rms < 1e-12) return 1e12;
    return wh_signal_peak(x, N) / rms;
}

double wh_signal_autocorrelation(const double* x, int N, int k) {
    if (!x || N <= k || k < 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < N - k; i++) {
        sum += x[i] * x[i + k];
    }
    return sum / (N - k);
}

/* ??? Multisine generation ??????????????????????????????????????????????? */

int wh_signal_multisine(double* out, int N, double fs,
                         double f_min, double f_max, int n_harmonics,
                         double amplitude, unsigned int seed) {
    if (!out || N <= 0 || fs <= 0.0 || f_min < 0.0 || f_max > fs / 2.0
        || n_harmonics <= 0) return -1;
    sig_srand(seed);
    memset(out, 0, N * sizeof(double));

    /* Frequency resolution */
    double df = fs / N;
    int k_min = (int)ceil(f_min / df);
    int k_max = (int)floor(f_max / df);
    if (k_max <= k_min) return -1;

    /* Choose n_harmonics frequencies uniformly spanning [k_min, k_max] */
    int n_available = k_max - k_min + 1;
    int stride = (n_available > n_harmonics) ? (n_available / n_harmonics) : 1;
    int count = 0;
    for (int k = k_min; k <= k_max && count < n_harmonics; k += stride) {
        double phase = 2.0 * M_PI * sig_rand();
        for (int i = 0; i < N; i++) {
            out[i] += (amplitude / sqrt((double)n_harmonics))
                      * sin(2.0 * M_PI * k * i / (double)N + phase);
        }
        count++;
    }
    return count;
}

int wh_signal_multisine_odd(double* out, int N, double fs,
                             double f_max, int n_harmonics,
                             double amplitude, unsigned int seed) {
    if (!out || N <= 0 || fs <= 0.0 || f_max > fs / 2.0 || n_harmonics <= 0)
        return -1;
    sig_srand(seed);
    memset(out, 0, N * sizeof(double));

    double df = fs / N;
    int k_max = (int)floor(f_max / df);
    int count = 0;

    /* Use odd harmonics only */
    for (int k = 1; k <= k_max && count < n_harmonics; k += 2) {
        /* Skip multiples of 3 to avoid exciting even nonlinear */
        if (k % 3 == 0) continue;
        double phase = 2.0 * M_PI * sig_rand();
        for (int i = 0; i < N; i++) {
            out[i] += (amplitude / sqrt((double)n_harmonics))
                      * sin(2.0 * M_PI * k * i / (double)N + phase);
        }
        count++;
    }
    return count;
}

int wh_signal_multisine_full(double* out, int N, double fs,
                              double f_min, double f_max,
                              double amplitude, unsigned int seed) {
    if (!out || N <= 0 || fs <= 0.0) return -1;
    sig_srand(seed);
    memset(out, 0, N * sizeof(double));

    double df = fs / N;
    int k_min = (int)ceil(f_min / df);
    int k_max = (int)floor(f_max / df);
    if (k_max <= k_min) return -1;

    int n_harmonics = k_max - k_min;
    for (int k = k_min; k <= k_max; k++) {
        double phase = 2.0 * M_PI * sig_rand();
        for (int i = 0; i < N; i++) {
            out[i] += (amplitude / sqrt((double)n_harmonics))
                      * sin(2.0 * M_PI * k * i / (double)N + phase);
        }
    }
    return n_harmonics;
}

/* ??? Chirp ?????????????????????????????????????????????????????????????? */

int wh_signal_chirp(double* out, int N, double fs,
                     double f0, double f1, double A) {
    if (!out || N <= 0 || fs <= 0.0) return -1;
    double T = N / fs;
    for (int i = 0; i < N; i++) {
        double t = i / fs;
        double phase = 2.0 * M_PI * (f0 * t + 0.5 * (f1 - f0) * t * t / T);
        out[i] = A * sin(phase);
    }
    return 0;
}

/* ??? PRBS ??????????????????????????????????????????????????????????????? */

int wh_signal_prbs(double* out, int N, double A, int n_stages,
                    unsigned int seed) {
    if (!out || N <= 0 || n_stages < 2 || n_stages > 31) return -1;

    /* Fibonacci LFSR with taps for maximal length sequences */
    static const int taps[] = {
        0, 0, 0x3, 0x6, 0xC, 0x14, 0x30, 0x60, 0xB8, 0x110, 0x240,
        0x500, 0xE08, 0x1C80, 0x3800, 0x6000, 0xD008, 0x12000, 0x24000,
        0x50000, 0xC0000, 0x180000, 0x300000, 0x600000, 0xD80000,
        0x1200000, 0x2400000, 0x5000000, 0xC000000, 0x18000000, 0x30000000
    };

    sig_srand(seed);
    unsigned int state = (unsigned int)(sig_rand() * 0x7FFFFFFF) | 1;
    unsigned int tap = (n_stages <= 31) ? taps[n_stages] : taps[31];

    for (int i = 0; i < N; i++) {
        out[i] = (state & 1) ? A : -A;
        /* Shift and feedback */
        unsigned int feedback = 0;
        unsigned int mask = tap;
        for (int b = 0; b < n_stages; b++) {
            if (mask & 1) feedback ^= (state >> b) & 1;
            mask >>= 1;
        }
        state = (state >> 1) | (feedback << (n_stages - 1));
    }
    return 0;
}

/* ??? Colored Gaussian (AR process) ?????????????????????????????????????? */

int wh_signal_arx(double* out, int N, const double* a, int order,
                   double sigma, unsigned int seed) {
    if (!out || N <= 0 || !a || order < 0 || order > 63) return -1;
    sig_srand(seed);

    /* Zero-initialize the output buffer */
    memset(out, 0, N * sizeof(double));

    for (int i = 0; i < N; i++) {
        double e = sigma * sig_randn();
        out[i] = e;
        for (int j = 1; j <= order && (i - j) >= 0; j++) {
            out[i] -= a[j] * out[i - j];
        }
        /* Note: a[0] is assumed to be 1.0 (monic) */
        if (fabs(a[0]) > 1e-12 && fabs(a[0] - 1.0) > 1e-12)
            out[i] /= a[0];
    }
    return 0;
}

/* ??? Gaussian noise ????????????????????????????????????????????????????? */

int wh_signal_gaussian(double* out, int N, double mean, double sigma,
                        unsigned int seed) {
    if (!out || N <= 0) return -1;
    sig_srand(seed);
    for (int i = 0; i < N; i++) {
        out[i] = mean + sigma * sig_randn();
    }
    return 0;
}

/* ??? Deterministic signals ?????????????????????????????????????????????? */

void wh_signal_step(double* out, int N, int k_step,
                     double u0, double amplitude) {
    if (!out || N <= 0) return;
    for (int i = 0; i < N; i++) {
        out[i] = (i < k_step) ? u0 : (u0 + amplitude);
    }
}

void wh_signal_sine(double* out, int N, double fs, double f,
                     double A, double phase, double offset) {
    if (!out || N <= 0) return;
    for (int i = 0; i < N; i++) {
        out[i] = offset + A * sin(2.0 * M_PI * f * i / fs + phase);
    }
}

void wh_signal_ramp(double* out, int N, double u0, double slope, double fs) {
    if (!out || N <= 0) return;
    for (int i = 0; i < N; i++) {
        out[i] = u0 + slope * i / fs;
    }
}

/* ??? Signal conditioning ???????????????????????????????????????????????? */

void wh_signal_normalize(double* in_out, int N) {
    if (!in_out || N <= 1) return;
    double mean = wh_signal_mean(in_out, N);
    double std = sqrt(wh_signal_variance(in_out, N));
    if (std < 1e-12) std = 1.0;
    for (int i = 0; i < N; i++) {
        in_out[i] = (in_out[i] - mean) / std;
    }
}

void wh_signal_detrend(double* in_out, int N) {
    if (!in_out || N <= 1) return;
    /* Fit y = a + b*i using linear regression */
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < N; i++) {
        sum_x += i;
        sum_y += in_out[i];
        sum_xy += i * in_out[i];
        sum_xx += (double)i * i;
    }
    double b = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x * sum_x);
    double a = (sum_y - b * sum_x) / N;

    for (int i = 0; i < N; i++) {
        in_out[i] -= (a + b * i);
    }
}

int wh_signal_downsample(const double* in, int n_in, double* out, int factor) {
    if (!in || !out || n_in <= 0 || factor < 1) return 0;
    int n_out = n_in / factor;
    for (int i = 0; i < n_out; i++) {
        double sum = 0.0;
        for (int j = 0; j < factor; j++) {
            sum += in[i * factor + j];
        }
        out[i] = sum / factor;
    }
    return n_out;
}

void wh_signal_filter_lp(double* in_out, int N, double fc, double fs) {
    if (!in_out || N <= 0 || fc <= 0.0 || fs <= 0.0) return;
    double alpha = 2.0 * M_PI * fc / fs;
    if (alpha > 1.0) alpha = 1.0;
    double y_prev = in_out[0];
    for (int i = 1; i < N; i++) {
        y_prev = alpha * in_out[i] + (1.0 - alpha) * y_prev;
        in_out[i] = y_prev;
    }
}
