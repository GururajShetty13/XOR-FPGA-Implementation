#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<16,4>  data_t;
typedef ap_fixed<16,4>  weight_t;
typedef ap_uint<3>      state_t;

const data_t LR = data_t(0.3);

enum State {
    S_IDLE    = 0,
    S_INIT    = 1,
    S_FORWARD = 2,
    S_PREDICT = 3,
    S_ERROR   = 4,
    S_UPDATE  = 5,
    S_DONE    = 6
};

static data_t sigmoid_approx(data_t x) {
#pragma HLS INLINE
    if (x < data_t(-4.0)) return data_t(0.0);
    if (x > data_t( 4.0)) return data_t(1.0);
    return data_t(0.5) + x * data_t(0.125);
}

static data_t sigmoid_derivative(data_t a) {
#pragma HLS INLINE
    return a * (data_t(1.0) - a);
}

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
) {
#pragma HLS INTERFACE ap_none port=rst_r
#pragma HLS INTERFACE ap_none port=start_r
#pragma HLS INTERFACE ap_none port=init_mode
#pragma HLS INTERFACE ap_none port=train_mode
#pragma HLS INTERFACE ap_none port=x1
#pragma HLS INTERFACE ap_none port=x2
#pragma HLS INTERFACE ap_none port=y_true
#pragma HLS INTERFACE ap_none port=y_out
#pragma HLS INTERFACE ap_none port=pred_out
#pragma HLS INTERFACE ap_none port=done
#pragma HLS INTERFACE ap_none port=dbg_w1_00
#pragma HLS INTERFACE ap_none port=dbg_w1_10
#pragma HLS INTERFACE ap_none port=dbg_b1_0
#pragma HLS INTERFACE ap_none port=dbg_w2_0
#pragma HLS INTERFACE ap_none port=dbg_b2
#pragma HLS INTERFACE ap_none port=dbg_phase
#pragma HLS INTERFACE ap_none port=dbg_phase_text
#pragma HLS INTERFACE ap_none port=dbg_sample_id
#pragma HLS INTERFACE ap_ctrl_none port=return

    const int H = 4;

    static state_t state = S_IDLE;
    static ap_uint<1> done_reg = 0;
    static data_t y_out_reg = 0;
    static ap_uint<1> pred_out_reg = 0;
    static state_t phase_reg = S_IDLE;
    static ap_uint<64> phase_text_reg = 0x49444C4520202020ULL; // "IDLE    "

    static weight_t W1[2][H];
    static weight_t b1[H];
    static weight_t W2[H];
    static weight_t b2;

    static data_t a1[H];
    static data_t d1[H];
    static data_t a2 = 0;
    static data_t d2 = 0;
    static data_t sum = 0;

#pragma HLS ARRAY_PARTITION variable=W1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=b1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=W2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=a1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=d1 complete dim=0

    if (x1 == data_t(0.0) && x2 == data_t(0.0)) dbg_sample_id = 0;
    else if (x1 == data_t(0.0) && x2 == data_t(1.0)) dbg_sample_id = 1;
    else if (x1 == data_t(1.0) && x2 == data_t(0.0)) dbg_sample_id = 2;
    else dbg_sample_id = 3;

    if (rst_r) {
        state = S_IDLE;
        done_reg = 0;
        y_out_reg = 0;
        pred_out_reg = 0;
        phase_reg = S_IDLE;
        phase_text_reg = 0x49444C4520202020ULL; // "IDLE    "

        for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
            for (int j = 0; j < H; j++) {
#pragma HLS UNROLL
                W1[i][j] = 0;
            }
        }

        for (int j = 0; j < H; j++) {
#pragma HLS UNROLL
            b1[j] = 0;
            W2[j] = 0;
            a1[j] = 0;
            d1[j] = 0;
        }

        b2 = 0;
        a2 = 0;
        d2 = 0;
        sum = 0;
    } else {
        switch (state) {
        case S_IDLE:
            phase_reg = S_IDLE;
            phase_text_reg = 0x49444C4520202020ULL; // "IDLE    "
            done_reg = 0;
            if (start_r) {
                if (init_mode) {
                    state = S_INIT;
                } else {
                    state = S_FORWARD;
                }
            }
            break;

        case S_INIT:
            phase_reg = S_INIT;
            phase_text_reg = 0x494E495420202020ULL; // "INIT    "

            // Known fixed weights for verification
            W1[0][0] = weight_t(0.2);
            W1[0][1] = weight_t(-0.1);
            W1[0][2] = weight_t(0.4);
            W1[0][3] = weight_t(0.1);

            W1[1][0] = weight_t(-0.3);
            W1[1][1] = weight_t(0.2);
            W1[1][2] = weight_t(0.1);
            W1[1][3] = weight_t(-0.2);

            b1[0] = weight_t(0.0);
            b1[1] = weight_t(0.0);
            b1[2] = weight_t(0.0);
            b1[3] = weight_t(0.0);

            W2[0] = weight_t(0.3);
            W2[1] = weight_t(-0.4);
            W2[2] = weight_t(0.2);
            W2[3] = weight_t(0.1);

            b2 = weight_t(0.05);

            state = S_DONE;
            break;

        case S_FORWARD:
            phase_reg = S_FORWARD;
            phase_text_reg = 0x464F525741524420ULL; // "FORWARD "
            sum = b2;

            for (int j = 0; j < H; j++) {
#pragma HLS UNROLL
                data_t mul1 = x1 * W1[0][j];
                data_t mul2 = x2 * W1[1][j];
                data_t z    = mul1 + mul2 + b1[j];
                data_t a    = sigmoid_approx(z);
                a1[j]       = a;
                data_t prod = a * W2[j];
                sum         = sum + prod;
            }

            state = S_PREDICT;
            break;

        case S_PREDICT:
            phase_reg = S_PREDICT;
            phase_text_reg = 0x5052454449435420ULL; // "PREDICT "
            a2 = sigmoid_approx(sum);

            if (a2 < data_t(0.0)) a2 = data_t(0.0);
            if (a2 > data_t(1.0)) a2 = data_t(1.0);

            y_out_reg = a2;
            pred_out_reg = (a2 > data_t(0.5)) ? ap_uint<1>(1) : ap_uint<1>(0);

            if (train_mode) {
                state = S_ERROR;
            } else {
                state = S_DONE;
            }
            break;

        case S_ERROR:
            phase_reg = S_ERROR;
            phase_text_reg = 0x4552524F52202020ULL; // "ERROR   "

            {
                data_t err = a2 - y_true;
                data_t sd2 = sigmoid_derivative(a2);
                d2 = err * sd2;
            }

            for (int j = 0; j < H; j++) {
#pragma HLS UNROLL
                data_t tmp1 = d2 * W2[j];
                data_t tmp2 = sigmoid_derivative(a1[j]);
                d1[j] = tmp1 * tmp2;
            }

            state = S_UPDATE;
            break;

        case S_UPDATE:
            phase_reg = S_UPDATE;
            phase_text_reg = 0x5550444154452020ULL; // "UPDATE  "

            for (int j = 0; j < H; j++) {
#pragma HLS UNROLL
                data_t grad = a1[j] * d2;
                data_t step = LR * grad;
                W2[j] = W2[j] - step;
            }

            {
                data_t step_b2 = LR * d2;
                b2 = b2 - step_b2;
            }

            for (int j = 0; j < H; j++) {
#pragma HLS UNROLL
                data_t grad_w10 = LR * (x1 * d1[j]);
                data_t grad_w11 = LR * (x2 * d1[j]);
                data_t grad_b1  = LR * d1[j];

                W1[0][j] = W1[0][j] - grad_w10;
                W1[1][j] = W1[1][j] - grad_w11;
                b1[j]    = b1[j]    - grad_b1;
            }

            state = S_DONE;
            break;

        case S_DONE:
            phase_reg = S_DONE;
            phase_text_reg = 0x444F4E4520202020ULL; // "DONE    "
            done_reg = 1;
            state = S_IDLE;
            break;

        default:
            state = S_IDLE;
            phase_reg = S_IDLE;
            phase_text_reg = 0x3F3F3F3F3F3F3F3FULL; // "????????"
            done_reg = 0;
            break;
        }
    }

    y_out = y_out_reg;
    pred_out = pred_out_reg;
    done = done_reg;
    dbg_phase = phase_reg;
    dbg_phase_text = phase_text_reg;

    dbg_w1_00 = W1[0][0];
    dbg_w1_10 = W1[1][0];
    dbg_b1_0  = b1[0];
    dbg_w2_0  = W2[0];
    dbg_b2    = b2;
}