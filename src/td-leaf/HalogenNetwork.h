#pragma once

#include "../BitBoardDefine.h"

constexpr std::array architecture = {
    768,
    16,
    1
};

class Position;

// It's more efficent to incrementally update using weights[input][output]
template <typename T, size_t in_count, size_t out_count>
struct TransposeLayer
{
    constexpr static size_t inputs = in_count;
    constexpr static size_t outputs = out_count;

    std::array<std::array<T, out_count>, in_count> weight;
    std::array<T, out_count> bias;
};

// when doing matrix/vector multiplications, it's more efficent to have weights[output][input]
template <typename T, size_t in_count, size_t out_count>
struct Layer
{
    constexpr static size_t inputs = in_count;
    constexpr static size_t outputs = out_count;

    std::array<std::array<T, in_count>, out_count> weight;
    std::array<T, out_count> bias;
};

// this is the minimum interface required for the rest of Halogen code to accept the network
class HalogenNetwork
{
public:
    void Recalculate(const Position& position);

    // calculates starting from the first hidden layer and skips input -> hidden
    int16_t Eval(Players stm) const;

    // call and then update inputs as required
    void AccumulatorPush();

    void AddInput(Square square, Pieces piece);
    void RemoveInput(Square square, Pieces piece);

    // do undo the last move
    void AccumulatorPop();

protected:
    int index(Square square, Pieces piece) const;

    static TransposeLayer<float, architecture[0], architecture[1]> l1;
    static Layer<float, architecture[1], architecture[2]> l2;

private:
    using Accumulator = std::array<float, architecture[1]>;

    std::vector<Accumulator> AccumulatorStack;
};
