#pragma once

#include "HalogenNetwork.h"
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>

class TrainableNetwork : public HalogenNetwork
{
public:
    static std::array<std::vector<int>, N_PLAYERS> GetSparseInputs(const BoardState& position);

    static void InitializeWeightsRandomly(bool print_diagnostics = false);
    static void SaveWeights(const std::string& filename, bool print_diagnostics = false);
    static void PrintNetworkDiagnostics();
    static bool VerifyWeightReadWrite();

    void UpdateGradients(double loss_gradient, const std::array<std::vector<int>, N_PLAYERS>& sparse_inputs, Players stm);
    void ApplyOptimizationStep(int n_samples);

    struct adam_state
    {
        float m = 0;
        float v = 0;

        constexpr static double beta_1 = 0.9;
        constexpr static double beta_2 = 0.999;
        constexpr static double epsilon = 10e-8;
        static double alpha;
    };

private:
    static std::mutex l1_lock;
    static std::mutex l2_lock;

    static TransposeLayer<adam_state, architecture[0], architecture[1]> l1_adam;
    static Layer<adam_state, architecture[1] * 2, architecture[2]> l2_adam;

    // for bias adjustment
    static std::atomic<uint64_t> t;

    TransposeLayer<float, architecture[0], architecture[1]> l1_gradient = {};
    Layer<float, architecture[1] * 2, architecture[2]> l2_gradient = {};
};

// t ranges from 0 at the begining to 1 at the end
inline double learning_rate_schedule(double t)
{
    // cosine annealing
    static constexpr double initial_lr = 0.001;
    return initial_lr * (cos(t * M_PI) + 1.0) / 2.0;
}

// The current adjusted learning rate.
inline double TrainableNetwork::adam_state::alpha = learning_rate_schedule(0);