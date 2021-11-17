#include "Network.h"
#include "Position.h"
#include "incbin/incbin.h"

INCBIN(Net, EVALFILE);

std::array<float, HALF_L1> Network::l1_bias;
std::array<std::array<float, HALF_L1>, INPUT_NEURONS> Network::l1_weight;

std::array<float, L2_NEURONS> Network::l2_bias;
std::array<std::array<float, L2_NEURONS>, L1_NEURONS> Network::l2_weight;

std::array<float, L3_NEURONS> Network::l3_bias;
std::array<std::array<float, L3_NEURONS>, L2_NEURONS> Network::l3_weight;

std::array<float, OUTPUT_NEURONS> Network::out_bias;
std::array<std::array<float, OUTPUT_NEURONS>, L3_NEURONS> Network::out_weight;

template <typename T, size_t SIZE>
void AddToUs(std::array<T, SIZE * 2>& a, const std::array<T, SIZE>& b)
{
    for (size_t i = 0; i < SIZE; i++)
        a[i] += b[i];
}

template <typename T, size_t SIZE>
void AddToThem(std::array<T, SIZE * 2>& a, const std::array<T, SIZE>& b)
{
    for (size_t i = 0; i < SIZE; i++)
        a[i + SIZE] += b[i];
}

template <typename T, size_t SIZE_A, size_t SIZE_B>
void MultiplyVectorByMatrix(const std::array<T, SIZE_A>& vec, const std::array<std::array<T, SIZE_B>, SIZE_A>& matrix, std::array<T, SIZE_B>& output)
{
    for (size_t i = 0; i < SIZE_A; i++)
        for (size_t j = 0; j < SIZE_B; j++)
            output[j] += vec[i] * matrix[i][j];
}

template <typename T, size_t SIZE>
void ReLU(std::array<T, SIZE>& source)
{
    for (size_t i = 0; i < SIZE; i++)
        source[i] = std::max(T(0), source[i]);
}

template <typename T, size_t SIZE>
void Softmax(std::array<T, SIZE>& source)
{
    // Get the max output
    T largest = *std::max_element(source.begin(), source.end());

    // Calculate exp(x - largest)
    std::transform(source.begin(), source.end(), source.begin(), [largest](T& x)
        { return std::exp(x - largest); });

    // Calculate sum
    T sum = std::accumulate(source.begin(), source.end(), T(0));

    // The softmax is exp(x - max(x)) / sum
    std::transform(source.begin(), source.end(), source.begin(), [sum](T& x)
        { return x / sum; });
}

void Network::Init()
{
    auto data = reinterpret_cast<const float*>(gNetData);

    for (auto& val : l1_bias)
        val = *data++;

    for (auto& row : l1_weight)
        for (auto& val : row)
            val = *data++;

    for (auto& val : l2_bias)
        val = *data++;

    for (auto& row : l2_weight)
        for (auto& val : row)
            val = *data++;

    for (auto& val : l3_bias)
        val = *data++;

    for (auto& row : l3_weight)
        for (auto& val : row)
            val = *data++;

    for (auto& val : out_bias)
        val = *data++;

    for (auto& row : out_weight)
        for (auto& val : row)
            val = *data++;

    if (reinterpret_cast<const unsigned char*>(data) - gNetData != gNetSize)
    {
        std::cout << "Error! Network architecture is incompatable" << std::endl;
        throw;
    }
}

Rank RelativeRank(Players colour, Square sq)
{
    return colour == WHITE ? GetRank(sq) : static_cast<Rank>(RANK_8 - GetRank(sq));
}

Square RelativeSquare(Players colour, Square sq)
{
    return GetPosition(GetFile(sq), RelativeRank(colour, sq));
}

WDL Network::EvalWDL(const Position& position) const
{
    //------------------

    std::array<float, L1_NEURONS> l1 = {};
    AddToUs(l1, l1_bias);
    AddToThem(l1, l1_bias);

    Players stm = position.GetTurn();

    Square ourKing = RelativeSquare(stm, position.GetKing(stm));
    Square theirKing = RelativeSquare(!stm, position.GetKing(!stm));

    uint64_t nonKingMaterial = position.GetAllPieces() ^ position.GetPieceBB<KING>();

    while (nonKingMaterial)
    {
        Square sq = static_cast<Square>(LSBpop(nonKingMaterial));
        Pieces piece = position.GetSquare(sq);

        size_t inputUs = (64 * 10 * ourKing) + (64 * (5 * (ColourOfPiece(piece) == stm) + GetPieceType(piece))) + RelativeSquare(stm, sq);
        size_t inputThem = (64 * 10 * theirKing) + (64 * (5 * (ColourOfPiece(piece) != stm) + GetPieceType(piece))) + RelativeSquare(!stm, sq);

        AddToUs(l1, l1_weight[inputUs]);
        AddToThem(l1, l1_weight[inputThem]);
    }

    ReLU(l1);

    //------------------

    std::array<float, L2_NEURONS> l2 = l2_bias;

    MultiplyVectorByMatrix(l1, l2_weight, l2);
    ReLU(l2);

    //------------------

    std::array<float, L3_NEURONS> l3 = l3_bias;

    MultiplyVectorByMatrix(l2, l3_weight, l3);
    ReLU(l3);

    //------------------

    std::array<float, OUTPUT_NEURONS> output = out_bias;

    MultiplyVectorByMatrix(l3, out_weight, output);
    Softmax(output);

    //------------------

    if (position.GetTurn() == WHITE)
        return { output[0], output[1], output[2] };
    else
        return { output[2], output[1], output[0] };
}

int16_t Network::Eval(const Position& position) const
{
    auto score = EvalWDL(position);
    return static_cast<int16_t>(std::round(score.ToCP()));
}

float WDL::ToCP()
{
    /*
    First we calculate the 'expected' score as W + 0.5D
    Then we calculate the logit of this score, aka an inverse sigmoid.
    The answer is then scaled by 150 times which was empirically adjusted
    to maximize elo
    */

    constexpr static float epsilon = 1e-8;
    constexpr static float scale = 300;

    float expectation = std::clamp(win + 0.5f * draw, epsilon, 1 - epsilon);
    float eval = std::log(expectation / (1 - expectation));

    return scale * eval;
}