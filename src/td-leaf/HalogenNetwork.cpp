#include "HalogenNetwork.h"

#include "../Position.h"

#include "matrix_operations.h"

std::array<std::array<float, architecture[1]>, architecture[0]> HalogenNetwork::l1_weight;
std::array<float, architecture[1]> HalogenNetwork::l1_bias;

void HalogenNetwork::Recalculate(const Position& position)
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

void HalogenNetwork::AccumulatorPush()
{
    AccumulatorStack.push_back(AccumulatorStack.back());
}

void HalogenNetwork::AccumulatorPop()
{
    AccumulatorStack.pop_back();
}

void HalogenNetwork::AddInput(Square square, Pieces piece)
{
    size_t input = index(square, piece);

    for (size_t j = 0; j < architecture[1]; j++)
    {
        AccumulatorStack.back()[j] += l1_weight[input][j];
    }
}

void HalogenNetwork::RemoveInput(Square square, Pieces piece)
{
    size_t input = index(square, piece);

    for (size_t j = 0; j < architecture[1]; j++)
    {
        AccumulatorStack.back()[j] -= l1_weight[input][j];
    }
}

int16_t HalogenNetwork::Eval(Players) const
{
    return AccumulatorStack.back()[0];
}

int HalogenNetwork::index(Square square, Pieces piece) const
{
    return piece * 64 + square;
}