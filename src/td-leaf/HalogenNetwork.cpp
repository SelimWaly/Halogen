#include "HalogenNetwork.h"

#include "../Position.h"

#include "matrix_operations.h"

TransposeLayer<float, architecture[0], architecture[1]> HalogenNetwork::l1;
Layer<float, architecture[1], architecture[2]> HalogenNetwork::l2;

void HalogenNetwork::Recalculate(const Position& position)
{
    AccumulatorStack = { { l1.bias } };

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
        AccumulatorStack.back()[j] += l1.weight[input][j];
    }
}

void HalogenNetwork::RemoveInput(Square square, Pieces piece)
{
    size_t input = index(square, piece);

    for (size_t j = 0; j < architecture[1]; j++)
    {
        AccumulatorStack.back()[j] -= l1.weight[input][j];
    }
}

int16_t HalogenNetwork::Eval(Players) const
{
    return l2.bias[0] + dot_product<float>(copy_ReLU(AccumulatorStack.back()), l2.weight[0]);
}

int HalogenNetwork::index(Square square, Pieces piece) const
{
    return piece * 64 + square;
}