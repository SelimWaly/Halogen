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
};