#include "TrainableNetwork.h"
#include "../Position.h"
#include <fstream>
#include <random>

std::vector<int> TrainableNetwork::GetSparseInputs(const Position& position) const
{
    // this should closely match the implementation of the HalogenNetwork::Recalculate() function

    std::vector<int> sparseInputs;
    sparseInputs.reserve(32);

    for (int i = 0; i < N_PIECES; i++)
    {
        Pieces piece = static_cast<Pieces>(i);
        uint64_t bb = position.GetPieceBB(piece);

        while (bb)
        {
            Square sq = static_cast<Square>(LSBpop(bb));
            sparseInputs.push_back(index(sq, piece));
        }
    }

    return sparseInputs;
}

void TrainableNetwork::InitializeWeightsRandomly() const
{
    std::mt19937 gen(0);
    std::normal_distribution<float> dis(0, 0);

    for (auto& row : l1_weight)
    {
        for (auto& weight : row)
        {
            weight = dis(gen);
        }
    }

    for (auto& bias : l1_bias)
    {
        bias = dis(gen);
    }
}

void TrainableNetwork::SaveWeights(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);

    file.write(reinterpret_cast<const char*>(l1_weight.data()), sizeof(l1_weight));
    file.write(reinterpret_cast<const char*>(l1_bias.data()), sizeof(l1_bias));
}
