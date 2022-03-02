#pragma once

#include "../BitBoardDefine.h"

constexpr std::array architecture = {
    768,
    1
};

class Position;

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

    static std::array<std::array<float, architecture[1]>, architecture[0]> l1_weight;
    static std::array<float, architecture[1]> l1_bias;

private:
    using Accumulator = std::array<float, architecture[1]>;

    std::vector<Accumulator> AccumulatorStack;
};
