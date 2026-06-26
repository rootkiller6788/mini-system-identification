#include "clid_types.h"
#include <stdio.h>
#include <assert.h>
int main(void) {
    printf("Hello from mini-closed-loop-identification\n");
    CLID_TransferFcn tf = clid_tf_alloc(2, 3, 1, 0.1);
    assert(tf.na == 2);
    assert(tf.nb == 3);
    assert(tf.a != NULL);
    printf("TF alloc OK: na=%d nb=%d\n", tf.na, tf.nb);
    clid_tf_free(&tf);
    assert(tf.a == NULL);
    printf("All tests passed!\n");
    return 0;
}
