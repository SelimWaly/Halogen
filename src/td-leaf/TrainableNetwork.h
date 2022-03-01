#pragma once

#include "HalogenNetwork.h"

class TrainableNetwork : public HalogenNetwork
{
public:
    std::vector<int> GetSparseInputs(const Position& position) const;

    void InitializeWeightsRandomly() const;
    void SaveWeights(const std::string& filename) const;

private:
};