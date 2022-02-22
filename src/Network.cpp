#include "Network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "Position.h"
#include "incbin/incbin.h"

INCBIN(Net, "46.nn");

std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> Network::hiddenWeights = {};
std::array<int16_t, HIDDEN_NEURONS> Network::hiddenBias = {};
std::array<int16_t, HIDDEN_NEURONS> Network::outputWeights = {};
int16_t Network::outputBias = {};

constexpr int16_t L1_SCALE = 64;
constexpr int16_t L2_SCALE = 256;
constexpr double SCALE_FACTOR = 1; // Found empirically to maximize elo

template <typename T, size_t SIZE>
[[nodiscard]] std::array<T, SIZE> ReLU(const std::array<T, SIZE>& source)
{
    std::array<T, SIZE> ret;

    for (size_t i = 0; i < SIZE; i++)
        ret[i] = std::max(T(0), source[i]);

    return ret;
}

template <typename T_out, typename T_in, size_t SIZE>
void DotProduct(const std::array<T_in, SIZE>& a, const std::array<T_in, SIZE>& b, T_out& output)
{
    for (size_t i = 0; i < SIZE; i++)
        output += a[i] * b[i];
}

void Network::Init()
{
    // Koi nets must skip 8 bytes
    auto Data = reinterpret_cast<const float*>(gNetData + 8);

    for (size_t i = 0; i < INPUT_NEURONS; i++)
        for (size_t j = 0; j < HIDDEN_NEURONS; j++)
            hiddenWeights[i][j] = (int16_t)round(*Data++ * L1_SCALE);

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        hiddenBias[i] = (int16_t)round(*Data++ * L1_SCALE);

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        outputWeights[i] = (int16_t)round(*Data++ * SCALE_FACTOR * L2_SCALE);

    outputBias = (int16_t)round(*Data++ * SCALE_FACTOR * L2_SCALE);
}

void Network::Recalculate(const Position& position)
{
    Zeta = { hiddenBias };

    for (int i = 0; i < N_PIECES; i++)
    {
        Pieces piece = static_cast<Pieces>(i);
        uint64_t bb = position.GetPieceBB(piece);

        while (bb)
        {
            Square sq = static_cast<Square>(LSBpop(bb));
            AddInput(sq, piece);
        }
    }
}

void Network::AccumulatorPush()
{
    Zeta.push_back(Zeta.back());
}

void Network::AccumulatorPop()
{
    Zeta.pop_back();
}

void Network::AddInput(Square square, Pieces piece)
{
    size_t index = piece * 64 + square;

    for (size_t j = 0; j < HIDDEN_NEURONS; j++)
        Zeta.back()[j] += hiddenWeights[index][j];
}

void Network::RemoveInput(Square square, Pieces piece)
{
    size_t index = piece * 64 + square;

    for (size_t j = 0; j < HIDDEN_NEURONS; j++)
        Zeta.back()[j] -= hiddenWeights[index][j];
}

int16_t Network::Eval() const
{
    int32_t output = outputBias * L1_SCALE;
    DotProduct(ReLU(Zeta.back()), outputWeights, output);
    return output / L1_SCALE / L2_SCALE;
}

/*void QuantizationAnalysis()
{
    auto Data = reinterpret_cast<float*>(label);

    float weight = 0;

    //hidden bias
    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        weight = std::max(weight, abs(*Data++));

    std::cout << weight << std::endl;
    weight = 0;

    //hidden weight
    for (size_t i = 0; i < INPUT_NEURONS; i++)
        for (size_t j = 0; j < HIDDEN_NEURONS; j++)
            weight = std::max(weight, abs(*Data++));

    std::cout << weight << std::endl;
    weight = 0;

    //output bias
    weight = std::max(weight, abs(*Data++));

    std::cout << weight << std::endl;
    weight = 0;

    //output weights
    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        weight = std::max(weight, abs(*Data++));

    std::cout << weight << std::endl;
    weight = 0;
}*/