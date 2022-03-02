#pragma once

#include "HalogenNetwork.h"
#include <mutex>

class TrainableNetwork : public HalogenNetwork
{
public:
    std::vector<int> GetSparseInputs(const Position& position) const;

    void InitializeWeightsRandomly() const;
    void SaveWeights(const std::string& filename) const;
    void Backpropagate(double loss_gradient, const std::vector<int>& sparse_inputs);
    void PrintNetworkDiagnostics() const;

private:
    mutable std::mutex mutex;

    constexpr static double beta_1 = 0.9;
    constexpr static double beta_2 = 0.999;
    constexpr static double alpha = 0.1;
    constexpr static double epsilon = 10e-8;

    static std::array<std::array<float, architecture[1]>, architecture[0]> l1_weight_m;
    static std::array<float, architecture[1]> l1_bias_m;

    static std::array<std::array<float, architecture[1]>, architecture[0]> l1_weight_v;
    static std::array<float, architecture[1]> l1_bias_v;
};