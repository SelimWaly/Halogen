#include "HalogenNetwork.h"

#include "../BoardState.h"

#include "matrix_operations.h"

decltype(HalogenNetwork::l1) HalogenNetwork::l1;
decltype(HalogenNetwork::l2) HalogenNetwork::l2;

void HalogenNetwork::Recalculate(const BoardState& position)
{
    AccumulatorStack = { { l1.bias, l1.bias } };

    for (int i = 0; i < N_PIECES; i++)
    {
        Pieces piece = static_cast<Pieces>(i);
        uint64_t bb = position.GetPieceBB(piece);

        while (bb)
        {
            Square sq = LSBpop(bb);
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
    size_t white_index = index(square, piece, WHITE);
    size_t black_index = index(square, piece, BLACK);

    for (size_t j = 0; j < architecture[1]; j++)
    {
        AccumulatorStack.back()[WHITE][j] += l1.weight[white_index][j];
        AccumulatorStack.back()[BLACK][j] += l1.weight[black_index][j];
    }
}

void HalogenNetwork::RemoveInput(Square square, Pieces piece)
{
    size_t white_index = index(square, piece, WHITE);
    size_t black_index = index(square, piece, BLACK);

    for (size_t j = 0; j < architecture[1]; j++)
    {
        AccumulatorStack.back()[WHITE][j] -= l1.weight[white_index][j];
        AccumulatorStack.back()[BLACK][j] -= l1.weight[black_index][j];
    }
}

int16_t HalogenNetwork::Eval(Players stm) const
{
    auto output = l2.bias[0] + dot_product_halves<decltype(l2)::value_type>(copy_ReLU(AccumulatorStack.back()[stm]), copy_ReLU(AccumulatorStack.back()[!stm]), l2.weight[0]);

    // 'half' or 'relative' nets return a score relative to the side to move
    // but Halogen expects a score relative to white
    return stm == WHITE ? output : -output;
}

Square MirrorVertically(Square sq)
{
    return static_cast<Square>(sq ^ 56);
}

int HalogenNetwork::index(Square square, Pieces piece, Players view)
{
    Square sq = view == WHITE ? square : MirrorVertically(square);
    Pieces relativeColor = static_cast<Pieces>(view == ColourOfPiece(piece));
    PieceTypes pieceType = GetPieceType(piece);

    return sq + pieceType * 64 + relativeColor * 64 * 6;
}