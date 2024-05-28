#pragma once

#include "../Score.h"

#include <cmath>

inline float learning_rate_schedule(float)
{
    static constexpr float initial_lr = 0.001;
    return initial_lr;
}

// The current adjusted learning rate.
inline float lr_alpha = learning_rate_schedule(0);

inline constexpr double training_time_hours = 24;

inline constexpr double sigmoid_coeff = 2.5 / 400;

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(sigmoid_coeff * -x));
}

inline float sigmoid_prime(float x)
{
    // note derivative of sigmoid s(v) with coefficent k is k*(s(v))*(1-s(v))
    auto s = sigmoid(x);
    return sigmoid_coeff * s * (1 - s);
}

inline constexpr auto training_threads = 1;

inline constexpr auto opening_book_usage_pct = 0.05;

inline constexpr auto opening_cutoff = Score(500);

inline constexpr auto search_depth = 4;
