#include "Network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include "BitBoardDefine.h"
#include "Position.h"
#include "incbin/incbin.h"

INCBIN(Net, EVALFILE);

std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> Network::hiddenWeights = {};
std::array<int16_t, HIDDEN_NEURONS> Network::hiddenBias = {};
std::array<int16_t, HIDDEN_NEURONS* 2> Network::outputWeights = {};
int16_t Network::outputBias = {};

auto calculate_quantization_l1()
{
    auto Data = reinterpret_cast<const float*>(gNetData);

    std::array<std::array<float, INPUT_NEURONS>, HIDDEN_NEURONS> quantization_l1_weight;
    std::array<float, HIDDEN_NEURONS> quantization_l1_bias;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        for (size_t j = 0; j < INPUT_NEURONS; j++)
            quantization_l1_weight[i][j] = *Data++;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        quantization_l1_bias[i] = *Data++;

    for (auto& row : quantization_l1_weight)
        std::sort(row.begin(), row.end());

    std::array<float, HIDDEN_NEURONS> highest = quantization_l1_bias;
    std::array<float, HIDDEN_NEURONS> lowest = quantization_l1_bias;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            highest[i] += quantization_l1_weight[i][INPUT_NEURONS - 1 - j];
            lowest[i] += quantization_l1_weight[i][j];
        }
    }

    float max_activation = *std::max_element(highest.begin(), highest.end());
    float min_activation = *std::min_element(lowest.begin(), lowest.end());

#ifndef NDEBUG
    std::cout << "max activation: " << max_activation << std::endl;
    std::cout << "min activation: " << min_activation << std::endl;
#endif

    auto quantization = std::numeric_limits<int16_t>::max() / std::max(std::abs(max_activation), std::abs(min_activation));
    int16_t limited_quantization = std::pow(2, std::floor(std::log2(quantization)));

#ifndef NDEBUG
    std::cout << "l1 quantization: " << quantization << std::endl;
    std::cout << "l1 limited quantization: " << limited_quantization << std::endl;
#endif

    return limited_quantization;
}

auto calculate_quantization_l2()
{
    auto Data = reinterpret_cast<const float*>(gNetData);
    Data += HIDDEN_NEURONS * (INPUT_NEURONS + 1);

    std::array<float, HIDDEN_NEURONS * 2 + 1> quantization_l2_weight_plus_bias;

    for (size_t i = 0; i < HIDDEN_NEURONS * 2; i++)
        quantization_l2_weight_plus_bias[i] = *Data++;

    quantization_l2_weight_plus_bias[HIDDEN_NEURONS * 2] = *Data++;

    float max = *std::max_element(quantization_l2_weight_plus_bias.begin(), quantization_l2_weight_plus_bias.end());
    float min = *std::min_element(quantization_l2_weight_plus_bias.begin(), quantization_l2_weight_plus_bias.end());

#ifndef NDEBUG
    std::cout << "l2 max: " << max << std::endl;
    std::cout << "l2 min: " << min << std::endl;
#endif

    auto quantization = std::numeric_limits<int16_t>::max() / std::max(std::abs(max), std::abs(min)) / sqrt(HIDDEN_NEURONS * 2);
    int16_t limited_quantization = std::pow(2, std::floor(std::log2(quantization)));

#ifndef NDEBUG
    std::cout << "l2 quantization: " << quantization << std::endl;
    std::cout << "l2 limited quantization: " << limited_quantization << std::endl;
#endif

    return limited_quantization;
}

int16_t l1_quantization = calculate_quantization_l1();
int16_t l2_quantization = calculate_quantization_l2();

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
    std::cout << EVALFILE " embeded in binary in with dynamic quantization { " << l1_quantization << ", " << l2_quantization << " }" << std::endl;
    auto Data = reinterpret_cast<const float*>(gNetData);

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        for (size_t j = 0; j < INPUT_NEURONS; j++)
            hiddenWeights[j][i] = static_cast<int16_t>(std::round(*Data++ * l1_quantization));

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        hiddenBias[i] = static_cast<int16_t>(std::round(*Data++ * l1_quantization));

    for (size_t i = 0; i < HIDDEN_NEURONS * 2; i++)
        outputWeights[i] = static_cast<int16_t>(std::round(*Data++ * l2_quantization));

    outputBias = static_cast<int16_t>(std::round(*Data++ * l2_quantization));

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
    int32_t output = outputBias * l1_quantization;
    DotProductHalves(ReLU(AccumulatorStack.back().side[stm]), ReLU(AccumulatorStack.back().side[!stm]), outputWeights, output);
    output /= l2_quantization * l1_quantization;

    // 'half' or 'relative' nets return a score relative to the side to move
    // but Halogen expects a score relative to white
    return stm == WHITE ? output : -output;
}
