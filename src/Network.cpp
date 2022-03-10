#include "Network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "BitBoardDefine.h"
#include "Position.h"
#include "incbin/incbin.h"

INCBIN(Net, "768-16-1_g327530.nn");

std::array<std::array<float, HIDDEN_NEURONS>, INPUT_NEURONS> Network::hiddenWeights = {};
std::array<float, HIDDEN_NEURONS> Network::hiddenBias = {};
std::array<float, HIDDEN_NEURONS* 2> Network::outputWeights = {};
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
void DotProductHalves(const std::array<T_in, SIZE>& stm, const std::array<T_in, SIZE>& other, const std::array<T_in, SIZE * 2>& weights, T_out& output)
{
    for (size_t i = 0; i < SIZE; i++)
    {
        output += stm[i] * weights[i];
    }

    for (size_t i = 0; i < SIZE; i++)
    {
        output += other[i] * weights[i + SIZE];
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

    for (size_t i = 0; i < HIDDEN_NEURONS * 2; i++)
        outputWeights[i] = *Data++;

    outputBias = *Data++;

    assert(reinterpret_cast<const unsigned char*>(Data) == gNetData + gNetSize);
}

void Network::Recalculate(const Position& position)
{
    AccumulatorStack = { { hiddenBias, hiddenBias } };

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

int index(Square square, Pieces piece, Players view)
{
    Square sq = view == WHITE ? square : MirrorVertically(square);
    Pieces relativeColor = static_cast<Pieces>(view == ColourOfPiece(piece));
    PieceTypes pieceType = GetPieceType(piece);

    return sq + pieceType * 64 + relativeColor * 64 * 6;
}

void Network::AddInput(Square square, Pieces piece)
{
    size_t white_index = index(square, piece, WHITE);
    size_t black_index = index(square, piece, BLACK);

    for (size_t j = 0; j < HIDDEN_NEURONS; j++)
    {
        AccumulatorStack.back().side[WHITE][j] += hiddenWeights[white_index][j];
        AccumulatorStack.back().side[BLACK][j] += hiddenWeights[black_index][j];
    }
}

void Network::RemoveInput(Square square, Pieces piece)
{
    size_t white_index = index(square, piece, WHITE);
    size_t black_index = index(square, piece, BLACK);

    for (size_t j = 0; j < HIDDEN_NEURONS; j++)
    {
        AccumulatorStack.back().side[WHITE][j] -= hiddenWeights[white_index][j];
        AccumulatorStack.back().side[BLACK][j] -= hiddenWeights[black_index][j];
    }
}

int16_t Network::Eval(Players stm) const
{
    float output = outputBias;
    DotProductHalves(ReLU(AccumulatorStack.back().side[stm]), ReLU(AccumulatorStack.back().side[!stm]), outputWeights, output);

    // 'half' or 'relative' nets return a score relative to the side to move
    // but Halogen expects a score relative to white
    return stm == WHITE ? output : -output;
}
