#include "Network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "BitBoardDefine.h"
#include "Position.h"
#include "incbin/incbin.h"

INCBIN(Net, "768-16-1_g4862915.nn");

std::array<std::array<float, HIDDEN_NEURONS>, INPUT_NEURONS> Network::hiddenWeights = {};
std::array<float, HIDDEN_NEURONS> Network::hiddenBias = {};
std::array<float, HIDDEN_NEURONS> Network::outputWeights = {};
float Network::outputBias = {};

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
void DotProduct(const std::array<T_in, SIZE>& stm, const std::array<T_in, SIZE>& weights, T_out& output)
{
    for (size_t i = 0; i < SIZE; i++)
    {
        output += stm[i] * weights[i];
    }
}

void Network::Init()
{
    auto Data = reinterpret_cast<const float*>(gNetData);

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        for (size_t j = 0; j < INPUT_NEURONS; j++)
            hiddenWeights[j][i] = *Data++;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        hiddenBias[i] = *Data++;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        outputWeights[i] = *Data++;

    outputBias = *Data++;

    assert(reinterpret_cast<const unsigned char*>(Data) == gNetData + gNetSize);
}

void Network::Recalculate(const Position& position)
{
    AccumulatorStack = { { hiddenBias } };

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
    AccumulatorStack.push_back(AccumulatorStack.back());
}

void Network::AccumulatorPop()
{
    AccumulatorStack.pop_back();
}

Square MirrorVertically(Square sq)
{
    return static_cast<Square>(sq ^ 56);
}

int index(Square square, Pieces piece)
{
    return square + piece * 64;
}

void Network::AddInput(Square square, Pieces piece)
{
    size_t input = index(square, piece);

    for (size_t j = 0; j < HIDDEN_NEURONS; j++)
    {
        AccumulatorStack.back().vals[j] += hiddenWeights[input][j];
    }
}

void Network::RemoveInput(Square square, Pieces piece)
{
    size_t input = index(square, piece);

    for (size_t j = 0; j < HIDDEN_NEURONS; j++)
    {
        AccumulatorStack.back().vals[j] -= hiddenWeights[input][j];
    }
}

int16_t Network::Eval(Players) const
{
    float output = outputBias;
    DotProduct(ReLU(AccumulatorStack.back().vals), outputWeights, output);
    return output;
}
