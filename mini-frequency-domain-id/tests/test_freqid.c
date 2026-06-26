#include "freqid_defs.h"
#include "freqid_spectrum.h"
#include "freqid_identify.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tr = 0, tp = 0;
#define T(n) do { tr++; printf("  TEST: %s ... ", n); } while(0)
#define P()   do { printf("PASS\n"); tp++; } while(0)

static void t_cpx(void) {
    T("complex_make");
    freqid_complex z = freqid_complex_make(3.0, 4.0);
    assert(fabs(creal(z) - 3.0) < 1e-10);
    assert(fabs(cimag(z) - 4.0) < 1e-10);
    P();

    T("complex_mag");
    assert(fabs(freqid_complex_mag(z) - 5.0) < 1e-10);
    P();

    T("complex_db");
    assert(fabs(freqid_complex_db(z) - 20.0*log10(5.0)) < 1e-6);
    P();

    T("complex_mul");
    freqid_complex a = freqid_complex_make(1.0, 2.0);
    freqid_complex b = freqid_complex_make(3.0, 4.0);
    freqid_complex p = freqid_complex_mul(a, b);
    assert(fabs(creal(p) - (-5.0)) < 1e-10);
    assert(fabs(cimag(p) - 10.0) < 1e-10);
    P();
}

static void t_fv(void) {
    T("freq_vector_linear");
    freqid_freq_vector fv = {0};
    assert(freqid_freq_vector_linear(&fv, 0.0, 100.0, 11) == 0);
    assert(fv.n == 11);
    assert(fabs(fv.w[5] - 50.0) < 1e-10);
    freqid_freq_vector_free(&fv);
    P();

    T("freq_vector_log");
    assert(freqid_freq_vector_log(&fv, 1.0, 100.0, 11) == 0);
    assert(fv.n == 11);
    freqid_freq_vector_free(&fv);
    P();

    T("freq_vector_null");
    assert(freqid_freq_vector_linear(NULL, 0.0, 1.0, 5) == -1);
    P();
}

static void t_tf(void) {
    T("tf_create_eval");
    freqid_transfer_function tf = {0};
    double num[] = {1.0};
    double den[] = {10.0, 1.0};  /* 10 + s => G(s)=1/(s+10), G(0)=0.1 */
    assert(freqid_tf_create(&tf, num, 0, den, 1, 0) == 0);
    freqid_complex G0 = freqid_tf_eval(&tf, 0.0);
    assert(fabs(creal(G0) - 0.1) < 1e-10);
    P();

    T("tf_null");
    assert(freqid_tf_create(&tf, NULL, 0, den, 1, 0) == -1);
    freqid_tf_free(&tf);
    P();
}

static void t_win(void) {
    T("window_hann");
    double *w = freqid_window_generate(FREQID_WIN_HANN, 128, 0.0);
    assert(w != NULL);
    assert(fabs(w[63] - 1.0) < 2e-4);  /* Hann peak at N=128 is ~0.99985 */
    free(w);
    P();

    T("window_null");
    w = freqid_window_generate(FREQID_WIN_RECTANGLE, 0, 0.0);
    assert(w == NULL);
    P();
}

static void t_dft(void) {
    T("dft_dc");
    double x[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    freqid_complex *X = NULL;
    assert(freqid_dft_real(x, 8, &X) == 0);
    assert(fabs(creal(X[0]) - 8.0) < 1e-10);
    free(X);
    P();

    T("dft_null");
    assert(freqid_dft_real(NULL, 8, &X) == -1);
    P();
}

int main(void) {
    printf("=== mini-frequency-domain-id Test Suite ===\n\n");
    t_cpx();
    t_fv();
    t_tf();
    t_win();
    t_dft();
    printf("\n=== Results: %d/%d tests passed ===\n", tp, tr);
    return (tp == tr) ? 0 : 1;
}
