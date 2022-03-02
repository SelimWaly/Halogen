#include "TrainableNetwork.h"
#include "../Position.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>

std::array<std::array<float, architecture[1]>, architecture[0]> TrainableNetwork::l1_weight_m;
std::array<float, architecture[1]> TrainableNetwork::l1_bias_m;

std::array<std::array<float, architecture[1]>, architecture[0]> TrainableNetwork::l1_weight_v;
std::array<float, architecture[1]> TrainableNetwork::l1_bias_v;

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
    std::lock_guard<std::mutex> lock(mutex);

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
    std::lock_guard<std::mutex> lock(mutex);

    std::ofstream file(filename, std::ios::out | std::ios::binary);

    file.write(reinterpret_cast<const char*>(l1_weight.data()), sizeof(l1_weight));
    file.write(reinterpret_cast<const char*>(l1_bias.data()), sizeof(l1_bias));
}

void TrainableNetwork::Backpropagate(double loss_gradient, const std::vector<int>& sparse_inputs)
{
    std::lock_guard<std::mutex> lock(mutex);

    for (size_t i = 0; i < l1_weight.size(); i++)
    {
        double g = std::find(sparse_inputs.begin(), sparse_inputs.end(), i) != sparse_inputs.end() ? loss_gradient : 0;

        l1_weight_m[i][0] = beta_1 * l1_weight_m[i][0] + (1 - beta_1) * g;
        l1_weight_v[i][0] = beta_2 * l1_weight_v[i][0] + (1 - beta_2) * g * g;

        l1_weight[i][0] += -alpha * l1_weight_m[i][0] / std::sqrt(l1_weight_v[i][0] + epsilon);
    }
    l1_bias_m[0] = beta_1 * l1_bias_m[0] + (1 - beta_1) * loss_gradient;
    l1_bias_v[0] = beta_2 * l1_bias_v[0] + (1 - beta_2) * loss_gradient * loss_gradient;

    l1_bias[0] += -alpha * l1_bias_m[0] / std::sqrt(l1_bias_v[0] + epsilon);
}

void TrainableNetwork::PrintNetworkDiagnostics() const
{
    std::lock_guard<std::mutex> lock(mutex);

    for (int i = 0; i < N_PIECES; i++)
    {
        float sum = 0;

        for (int j = 0; j < N_SQUARES; j++)
        {
            sum += l1_weight[i * 64 + j][0];

            std::cout << l1_weight[i * 64 + j][0] << " ";

            if (j % N_FILES == N_FILES - 1)
            {
                std::cout << std::endl;
            }
        }

        sum /= N_SQUARES;

        std::cout << "piece " << i << ": " << sum << std::endl;
        std::cout << std::endl;
    }

    std::cout << "bias: " << l1_bias[0] << std::endl;
}