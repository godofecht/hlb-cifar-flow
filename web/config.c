/* Emscripten-only configuration ABI. All ML remains in train.flow/infer.flow. */
#include <stdio.h>
#include <stdlib.h>

static int number(const char *key, int value) {
    char buffer[24];
    snprintf(buffer, sizeof buffer, "%d", value);
    return setenv(key, buffer, 1);
}

int flow_web_configure(int train_count, int eval_count, int width, int batch,
                       int epochs, int holdout, int dump_predictions) {
    if (train_count < 2 || train_count > 64 ||
        eval_count < 1 || eval_count > 32 ||
        width < 2 || width > 4 ||
        batch < 2 || batch > 8 ||
        epochs < 1 || epochs > 3 ||
        holdout < 0 || holdout >= train_count - 1) {
        return -1;
    }
    if (number("HLB_TRAIN_LIMIT", train_count) ||
        number("HLB_EVAL_LIMIT", eval_count) ||
        number("HLB_WIDTH", width) ||
        number("HLB_BATCH", batch) ||
        number("HLB_EPOCHS", epochs) ||
        number("HLB_VAL_HOLDOUT", holdout) ||
        number("HLB_WHITEN_EXAMPLES", train_count - holdout) ||
        number("HLB_WHITEN_STRIDE", 8) ||
        number("HLB_DUMP_PREDICTIONS", dump_predictions)) {
        return -1;
    }
    return 0;
}
