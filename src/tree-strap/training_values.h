#pragma once

#include "../Score.h"

#include <cmath>

inline float learning_rate_schedule(float)
{
    // TreeStrap(alpha-beta) suggests 5e-7 LR.
    static constexpr float initial_lr = 5e-7;
    return initial_lr;
}

// The current adjusted learning rate.
inline float lr_alpha = learning_rate_schedule(0);

inline constexpr double training_time_hours = 24;

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(2.5 / 400 * -x));
}

inline constexpr auto training_threads = 1;

inline constexpr auto opening_book_usage_pct = 0.05;

inline constexpr auto opening_cutoff = Score(500);

// Bootstrapping from Game Tree Search suggests depth 1
inline constexpr auto min_learning_depth = 1;

inline constexpr auto search_depth = 4;

// From the td-leaf experiments, there are various things to try for example
// - Fixed nodes search vs fixed depth search
// - multi-threading: issues with using TT?
// - opening filtering based on eval
// - sigmod MSE?
