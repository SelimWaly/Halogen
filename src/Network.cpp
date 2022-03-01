#include "Network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "BitBoardDefine.h"
#include "Position.h"
#include "incbin/incbin.h"

INCBIN(Net, "768-1_g55287.nn");

std::array<float, INPUT_NEURONS> Network::l1_weight = {};
float Network::l1_bias = {};

constexpr double SCALE_FACTOR = 1; // Found empirically to maximize elo

void Network::Init()
{
    auto Data = reinterpret_cast<const float*>(gNetData);

    for (size_t i = 0; i < INPUT_NEURONS; i++)
        l1_weight[i] = *Data++;

    l1_bias = *Data++;

    assert(reinterpret_cast<const unsigned char*>(Data) == gNetData + gNetSize);
}

void Network::Recalculate(const Position& position)
{
    AccumulatorStack = { { l1_bias } };

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

int index(Square square, Pieces piece, Players)
{
    return square + piece * 64;
}

void Network::AddInput(Square square, Pieces piece)
{
    size_t sq_index = index(square, piece, WHITE);

    AccumulatorStack.back().value += l1_weight[sq_index];
}

void Network::RemoveInput(Square square, Pieces piece)
{
    size_t sq_index = index(square, piece, WHITE);

    AccumulatorStack.back().value -= l1_weight[sq_index];
}

int16_t Network::Eval(Players) const
{
    return AccumulatorStack.back().value + l1_bias;
}
