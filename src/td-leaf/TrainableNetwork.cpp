#include "TrainableNetwork.h"
#include "../Position.h"
#include "HalogenNetwork.h"
#include "matrix_operations.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

// apply the adam gradients to the weight layer
template <template <typename, size_t, size_t> class layer_t, typename T, size_t in, size_t out>
void apply_gradient(layer_t<T, in, out>& layer, const layer_t<TrainableNetwork::adam_state, in, out>& adam);

// given a loss gradient of layer l, and an activation for layer l-1, update the adam states in layer l
template <typename T, size_t in, size_t out>
void update_gradient(Layer<TrainableNetwork::adam_state, in, out>& adam, const std::array<T, out>& gradient, const std::array<T, in>& activation);

template <typename T, size_t in, size_t out>
void update_gradient_sparse(TransposeLayer<TrainableNetwork::adam_state, in, out>& adam, const std::array<T, out>& gradient, const std::vector<int>& activation);

template <typename T, size_t in, size_t out, typename Activation_prime>
std::array<T, in> calculate_loss_gradient(Layer<T, in, out>& layer, const std::array<T, out>& gradient, const std::array<T, in>& activation, Activation_prime&& activation_prime);

TransposeLayer<TrainableNetwork::adam_state, architecture[0], architecture[1]> TrainableNetwork::l1_adam;
Layer<TrainableNetwork::adam_state, architecture[1], architecture[2]> TrainableNetwork::l2_adam;

std::recursive_mutex TrainableNetwork::mutex;

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
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::mt19937 gen(0);

    auto initialize = [&gen](auto&& layer)
    {
        // Kaiming He initialization
        std::normal_distribution<float> dis(0, sqrt(2.0 / layer.inputs));

        for (auto& row : layer.weight)
        {
            for (auto& element : row)
            {
                element = dis(gen);
            }
        }
    };

    initialize(l1);
    initialize(l2);

    PrintNetworkDiagnostics();
}

void TrainableNetwork::SaveWeights(const std::string& filename) const
{
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::ofstream file(filename, std::ios::out | std::ios::binary);

    auto save_bias = [&file](auto& layer)
    {
        for (size_t i = 0; i < layer.bias.size(); i++)
        {
            file.write(reinterpret_cast<const char*>(&layer.bias[i]), sizeof(float));
        }
    };

    auto save_layer = [&file, &save_bias](auto& layer)
    {
        for (size_t i = 0; i < layer.outputs; i++)
        {
            for (size_t j = 0; j < layer.inputs; j++)
            {
                file.write(reinterpret_cast<const char*>(&layer.weight[i][j]), sizeof(float));
            }
        }

        save_bias(layer);
    };

    auto save_transpose_layer = [&file, &save_bias](auto& layer)
    {
        for (size_t i = 0; i < layer.outputs; i++)
        {
            for (size_t j = 0; j < layer.inputs; j++)
            {
                file.write(reinterpret_cast<const char*>(&layer.weight[j][i]), sizeof(float));
            }
        }

        save_bias(layer);
    };

    save_transpose_layer(l1);
    save_layer(l2);
}

void TrainableNetwork::Backpropagate(double loss_gradient, const std::vector<int>& sparse_inputs)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);

    // do the forward pass and save the activations:

    std::array<float, architecture[1]> l1_activation = l1.bias;

    for (size_t i = 0; i < sparse_inputs.size(); i++)
    {
        for (int j = 0; j < architecture[1]; j++)
        {
            l1_activation[j] += l1.weight[sparse_inputs[i]][j];
        }
    }

    apply_ReLU(l1_activation);

    // get loss gradients
    auto l2_loss_gradient = std::array { static_cast<float>(loss_gradient) };
    auto l1_loss_gradient = calculate_loss_gradient(l2, l2_loss_gradient, l1_activation, [](float x)
        { return x > 0.0f; });

    // gradient updates
    update_gradient_sparse(l1_adam, l1_loss_gradient, sparse_inputs);
    update_gradient(l2_adam, l2_loss_gradient, l1_activation);

    // now actually apply the gradients
    apply_gradient(l1, l1_adam);
    apply_gradient(l2, l2_adam);
}

template <template <typename, size_t, size_t> class layer_t, typename T, size_t in, size_t out>
void apply_gradient(layer_t<T, in, out>& layer, const layer_t<TrainableNetwork::adam_state, in, out>& adam)
{
    for (size_t i = 0; i < layer.weight.size(); i++)
    {
        for (size_t j = 0; j < layer.weight[i].size(); j++)
        {
            layer.weight[i][j] += -TrainableNetwork::adam_state::alpha * adam.weight[i][j].m / std::sqrt(adam.weight[i][j].v + TrainableNetwork::adam_state::epsilon);
        }
    }

    for (size_t i = 0; i < layer.bias.size(); i++)
    {
        layer.bias[i] += -TrainableNetwork::adam_state::alpha * adam.bias[i].m / std::sqrt(adam.bias[i].v + TrainableNetwork::adam_state::epsilon);
    }
}

template <typename T, size_t in, size_t out>
void update_gradient(Layer<TrainableNetwork::adam_state, in, out>& adam, const std::array<T, out>& gradient, const std::array<T, in>& activation)
{
    for (size_t i = 0; i < adam.weight.size(); i++)
    {
        for (size_t j = 0; j < adam.weight[i].size(); j++)
        {
            double g = gradient[i] * activation[j];
            adam.weight[i][j].m = TrainableNetwork::adam_state::beta_1 * adam.weight[i][j].m + (1 - TrainableNetwork::adam_state::beta_1) * g;
            adam.weight[i][j].v = TrainableNetwork::adam_state::beta_2 * adam.weight[i][j].v + (1 - TrainableNetwork::adam_state::beta_2) * g * g;
        }

        double g = gradient[i];
        adam.bias[i].m = TrainableNetwork::adam_state::beta_1 * adam.bias[i].m + (1 - TrainableNetwork::adam_state::beta_1) * g;
        adam.bias[i].v = TrainableNetwork::adam_state::beta_2 * adam.bias[i].v + (1 - TrainableNetwork::adam_state::beta_2) * g * g;
    }
}

template <typename T, size_t in, size_t out>
void update_gradient_sparse(TransposeLayer<TrainableNetwork::adam_state, in, out>& adam, const std::array<T, out>& gradient, const std::vector<int>& sparse_inputs)
{
    for (size_t i = 0; i < adam.weight.size(); i++)
    {
        double activation = std::find(sparse_inputs.begin(), sparse_inputs.end(), i) != sparse_inputs.end() ? 1 : 0;

        for (size_t j = 0; j < adam.weight[i].size(); j++)
        {
            double g = gradient[j] * activation;
            adam.weight[i][j].m = TrainableNetwork::adam_state::beta_1 * adam.weight[i][j].m + (1 - TrainableNetwork::adam_state::beta_1) * g;
            adam.weight[i][j].v = TrainableNetwork::adam_state::beta_2 * adam.weight[i][j].v + (1 - TrainableNetwork::adam_state::beta_2) * g * g;
        }
    }

    for (size_t i = 0; i < adam.bias.size(); i++)
    {
        double g = gradient[i];
        adam.bias[i].m = TrainableNetwork::adam_state::beta_1 * adam.bias[i].m + (1 - TrainableNetwork::adam_state::beta_1) * g;
        adam.bias[i].v = TrainableNetwork::adam_state::beta_2 * adam.bias[i].v + (1 - TrainableNetwork::adam_state::beta_2) * g * g;
    }
}

template <typename T, size_t in, size_t out, typename Activation_prime>
std::array<T, in> calculate_loss_gradient(Layer<T, in, out>& layer, const std::array<T, out>& gradient, const std::array<T, in>& activation, Activation_prime&& activation_prime)
{
    std::array<T, in> loss_gradient = {};

    for (size_t i = 0; i < layer.weight.size(); i++)
    {
        for (size_t j = 0; j < layer.weight[i].size(); j++)
        {
            loss_gradient[j] += gradient[i] * layer.weight[i][j];
        }
    }

    for (size_t i = 0; i < loss_gradient.size(); i++)
    {
        loss_gradient[i] *= activation_prime(activation[i]);
    }

    return loss_gradient;
}

void TrainableNetwork::PrintNetworkDiagnostics() const
{
    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto print_layer = [&](auto&& layer)
    {
        for (size_t i = 0; i < layer.weight.size(); i++)
        {
            for (size_t j = 0; j < layer.weight[i].size(); j++)
            {
                std::cout << layer.weight[i][j] << " ";
            }

            std::cout << std::endl;
        }

        std::cout << std::endl;

        for (size_t i = 0; i < layer.bias.size(); i++)
        {
            std::cout << layer.bias[i] << " ";
        }

        std::cout << std::endl;
        std::cout << std::endl;
    };

    print_layer(l1);
    print_layer(l2);
}
