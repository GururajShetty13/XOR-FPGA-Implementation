#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<16,4>  data_t;
typedef ap_fixed<16,4>  weight_t;
typedef ap_uint<3>      state_t;

void hls_f7(
    ap_uint<1> rst_r,
    ap_uint<1> start_r,
    ap_uint<1> init_mode,
    ap_uint<1> train_mode,
    data_t x1,
    data_t x2,
    data_t y_true,
    data_t &y_out,
    ap_uint<1> &pred_out,
    ap_uint<1> &done,
    weight_t &dbg_w1_00,
    weight_t &dbg_w1_10,
    weight_t &dbg_b1_0,
    weight_t &dbg_w2_0,
    weight_t &dbg_b2,
    state_t &dbg_phase,
    ap_uint<64> &dbg_phase_text,
    ap_uint<2> &dbg_sample_id
);

static void print_phase_text(ap_uint<64> v) {
    for (int i = 7; i >= 0; --i) {
        unsigned char c = (unsigned char)((v >> (i * 8)) & 0xFF);
        std::cout << (char)c;
    }
}

static void apply_reset(
    data_t &y_out,
    ap_uint<1> &pred_out,
    ap_uint<1> &done,
    weight_t &dbg_w1_00,
    weight_t &dbg_w1_10,
    weight_t &dbg_b1_0,
    weight_t &dbg_w2_0,
    weight_t &dbg_b2,
    state_t &dbg_phase,
    ap_uint<64> &dbg_phase_text,
    ap_uint<2> &dbg_sample_id
) {
    for (int i = 0; i < 4; i++) {
        hls_f7(
            1, 0, 0, 0,
            0, 0, 0,
            y_out, pred_out, done,
            dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
            dbg_phase, dbg_phase_text, dbg_sample_id
        );
    }

    hls_f7(
        0, 0, 0, 0,
        0, 0, 0,
        y_out, pred_out, done,
        dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
        dbg_phase, dbg_phase_text, dbg_sample_id
    );
}

static void step_until_done(
    ap_uint<1> init_mode,
    ap_uint<1> train_mode,
    data_t x1,
    data_t x2,
    data_t y_true,
    data_t &y_out,
    ap_uint<1> &pred_out,
    ap_uint<1> &done,
    weight_t &dbg_w1_00,
    weight_t &dbg_w1_10,
    weight_t &dbg_b1_0,
    weight_t &dbg_w2_0,
    weight_t &dbg_b2,
    state_t &dbg_phase,
    ap_uint<64> &dbg_phase_text,
    ap_uint<2> &dbg_sample_id
) {
    hls_f7(
        0, 1, init_mode, train_mode,
        x1, x2, y_true,
        y_out, pred_out, done,
        dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
        dbg_phase, dbg_phase_text, dbg_sample_id
    );

    int timeout = 0;
    while (done == 0 && timeout < 200) {
        hls_f7(
            0, 0, init_mode, train_mode,
            x1, x2, y_true,
            y_out, pred_out, done,
            dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
            dbg_phase, dbg_phase_text, dbg_sample_id
        );
        timeout++;
    }

    if (timeout >= 200) {
        std::cout << "Timeout waiting for done. phase="
                  << (int)dbg_phase << " phase_text=";
        print_phase_text(dbg_phase_text);
        std::cout << " sample_id=" << (int)dbg_sample_id << "\n";
    }

    hls_f7(
        0, 0, init_mode, train_mode,
        x1, x2, y_true,
        y_out, pred_out, done,
        dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
        dbg_phase, dbg_phase_text, dbg_sample_id
    );
}

int main() {
    data_t y_out = 0;
    ap_uint<1> pred_out = 0;
    ap_uint<1> done = 0;

    weight_t dbg_w1_00 = 0;
    weight_t dbg_w1_10 = 0;
    weight_t dbg_b1_0  = 0;
    weight_t dbg_w2_0  = 0;
    weight_t dbg_b2    = 0;

    state_t dbg_phase = 0;
    ap_uint<64> dbg_phase_text = 0;
    ap_uint<2> dbg_sample_id = 0;

    const int epochs = 6000;

    data_t inputs[4][2] = {
        {data_t(0.0), data_t(0.0)},
        {data_t(0.0), data_t(1.0)},
        {data_t(1.0), data_t(0.0)},
        {data_t(1.0), data_t(1.0)}
    };

    data_t labels[4] = {
        data_t(0.0),
        data_t(1.0),
        data_t(1.0),
        data_t(0.0)
    };

    int expected[4] = {0, 1, 1, 0};

    apply_reset(
        y_out, pred_out, done,
        dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
        dbg_phase, dbg_phase_text, dbg_sample_id
    );

    step_until_done(
        1, 0,
        data_t(0.0), data_t(0.0), data_t(0.0),
        y_out, pred_out, done,
        dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
        dbg_phase, dbg_phase_text, dbg_sample_id
    );

    std::cout << "After INIT:\n";
    std::cout << "W1_00=" << dbg_w1_00.to_float()
              << " W1_10=" << dbg_w1_10.to_float()
              << " b1_0=" << dbg_b1_0.to_float()
              << " W2_0=" << dbg_w2_0.to_float()
              << " b2=" << dbg_b2.to_float()
              << "\n\n";

    std::cout << "Initial inference:\n";
    for (int i = 0; i < 4; i++) {
        step_until_done(
            0, 0,
            inputs[i][0], inputs[i][1], data_t(0.0),
            y_out, pred_out, done,
            dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
            dbg_phase, dbg_phase_text, dbg_sample_id
        );

        std::cout << "INIT_TEST_" << i
                  << " x1=" << inputs[i][0].to_float()
                  << " x2=" << inputs[i][1].to_float()
                  << " y_out=" << y_out.to_float()
                  << " pred=" << (int)pred_out
                  << "\n";
    }

    std::cout << "\nTraining...\n";
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < 4; i++) {
            step_until_done(
                0, 1,
                inputs[i][0], inputs[i][1], labels[i],
                y_out, pred_out, done,
                dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
                dbg_phase, dbg_phase_text, dbg_sample_id
            );
        }

        if (e % 500 == 0) {
            std::cout << "Epoch " << e
                      << " W1_00=" << dbg_w1_00.to_float()
                      << " W1_10=" << dbg_w1_10.to_float()
                      << " b1_0=" << dbg_b1_0.to_float()
                      << " W2_0=" << dbg_w2_0.to_float()
                      << " b2=" << dbg_b2.to_float()
                      << "\n";
        }
    }

    std::cout << "\nFinal inference:\n";
    int errors = 0;
    for (int i = 0; i < 4; i++) {
        step_until_done(
            0, 0,
            inputs[i][0], inputs[i][1], data_t(0.0),
            y_out, pred_out, done,
            dbg_w1_00, dbg_w1_10, dbg_b1_0, dbg_w2_0, dbg_b2,
            dbg_phase, dbg_phase_text, dbg_sample_id
        );

        std::cout << "FINAL_TEST_" << i
                  << " x1=" << inputs[i][0].to_float()
                  << " x2=" << inputs[i][1].to_float()
                  << " y_out=" << y_out.to_float()
                  << " pred=" << (int)pred_out
                  << " expected=" << expected[i]
                  << "\n";

        if ((int)pred_out != expected[i]) {
            errors++;
        }
    }

    std::cout << "\nFinal visible weights:\n";
    std::cout << "W1_00=" << dbg_w1_00.to_float()
              << " W1_10=" << dbg_w1_10.to_float()
              << " b1_0=" << dbg_b1_0.to_float()
              << " W2_0=" << dbg_w2_0.to_float()
              << " b2=" << dbg_b2.to_float()
              << "\n";

    std::cout << "\nErrors = " << errors << "\n";
    return 0;
}