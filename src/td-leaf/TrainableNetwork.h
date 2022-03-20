#pragma once

#include "HalogenNetwork.h"
#include <array>
#include <mutex>

class TrainableNetwork : public HalogenNetwork
{
public:
    static std::array<std::vector<int>, N_PLAYERS> GetSparseInputs(const Position& position);

    static void InitializeWeightsRandomly();
    static void SaveWeights(const std::string& filename);
    static void PrintNetworkDiagnostics();

    void UpdateGradients(double loss_gradient, const std::array<std::vector<int>, N_PLAYERS>& sparse_inputs, Players stm);
    void ApplyOptimizationStep(int n_samples);

    struct adam_state
    {
        float m;
        float v;

        constexpr static double beta_1 = 0.9;
        constexpr static double beta_2 = 0.999;
        constexpr static double epsilon = 10e-8;
        static double alpha;
    };

private:
    static std::recursive_mutex mutex;

    static TransposeLayer<adam_state, architecture[0], architecture[1]> l1_adam;
    static Layer<adam_state, architecture[1] * 2, architecture[2]> l2_adam;

    TransposeLayer<float, architecture[0], architecture[1]> l1_gradient = {};
    Layer<float, architecture[1] * 2, architecture[2]> l2_gradient = {};
};

inline double TrainableNetwork::adam_state::alpha = 0.001 * 16;