#pragma once

#include "HalogenNetwork.h"
#include <array>
#include <mutex>

class TrainableNetwork : public HalogenNetwork
{
public:
    std::array<std::vector<int>, N_PLAYERS> GetSparseInputs(const Position& position) const;

    void InitializeWeightsRandomly() const;
    void SaveWeights(const std::string& filename) const;
    void Backpropagate(double loss_gradient, const std::array<std::vector<int>, N_PLAYERS>& sparse_inputs, Players stm);
    void PrintNetworkDiagnostics() const;

    struct adam_state
    {
        float m;
        float v;

        constexpr static double beta_1 = 0.9;
        constexpr static double beta_2 = 0.999;
        constexpr static double alpha = 0.001;
        constexpr static double epsilon = 10e-8;
    };

private:
    static std::recursive_mutex mutex;

    static TransposeLayer<adam_state, architecture[0], architecture[1]> l1_adam;
    static Layer<adam_state, architecture[1] * 2, architecture[2]> l2_adam;
};
