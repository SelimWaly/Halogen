#pragma once

#include <cmath>
#include <cstdint>

#include "GameState.h"
#include "Move.h"
#include "Score.h"

class SearchSharedState;

/*Tuneable search constants*/

inline double LMR_constant = -0.76;
constexpr double LMR_depth_coeff = 0.39;
constexpr double LMR_move_coeff = 0.12;
constexpr double LMR_depth_move_coeff = 0.67;

constexpr int Null_constant = 4;
constexpr int Null_depth_quotent = 6;
constexpr int Null_beta_quotent = 250;

constexpr int Futility_constant = 20;
constexpr int Futility_coeff = 82;
constexpr int Futility_depth = 15;

constexpr Score aspiration_window_mid_width = 15;

constexpr int Delta_margin = 200;

constexpr int SNMP_coeff = 119;
constexpr int SNMP_depth = 8;

constexpr int LMP_constant = 11;
constexpr int LMP_coeff = 7;
constexpr int LMP_depth = 6;

inline int History_quotent = 8192;

inline auto calculate_LMR_reduction()
{
    std::array<std::array<int, 64>, 64> ret = {};

    for (size_t i = 0; i < ret.size(); i++)
    {
        for (size_t j = 0; j < ret[i].size(); j++)
        {
            ret[i][j] = static_cast<int>(std::round(LMR_constant + LMR_depth_coeff * log(i + 1)
                + LMR_move_coeff * log(j + 1) + LMR_depth_move_coeff * log(i + 1) * log(j + 1)));
        }
    }

    return ret;
};

// [depth][move number]
inline std::array<std::array<int, 64>, 64> LMR_reduction = calculate_LMR_reduction();

/*----------------*/

struct SearchResult
{
    SearchResult(int score, Move move = Move::Uninitialized)
        : m_score(score)
        , m_move(move)
    {
    }

    SearchResult(Score score, Move move = Move::Uninitialized)
        : m_score(score)
        , m_move(move)
    {
    }

    Score GetScore() const
    {
        return m_score;
    }

    Move GetMove() const
    {
        return m_move;
    }

private:
    Score m_score;
    Move m_move;
};

void SearchThread(GameState& position, SearchSharedState& sharedData);