#pragma once

#include "Pyrrhic/tbprobe.h"
#include "MoveGenerator.h"
#include <ctime>
#include <algorithm>
#include <thread>
#include <cmath>

/*Tuneable search constants*/

inline double LMR_constant = -2.16;
inline double LMR_coeff = 1.18;

inline int Null_constant = 5;
inline int Null_depth_quotent = 7;
inline int Null_beta_quotent = 197;

inline int Futility_constant = 79;
inline int Futility_coeff = 54;
inline int Futility_depth = 9;

inline int Aspiration_window = 16;

inline int Delta_margin = 220;

inline int SNMP_coeff = 77;
inline int SNMP_depth = 6;

inline int LMP_constant = 8;
inline int LMP_coeff = 6;
inline int LMP_depth = 7;

/*----------------*/

struct SearchResult
{
	SearchResult(short score, Move move = Move()) : m_score(score), m_move(move) {}

	int GetScore() const { return m_score; }
	Move GetMove() const { return m_move; }

private:
	short m_score;
	Move m_move;
};

uint64_t SearchThread(Position position, SearchParameters parameters, const SearchLimits& limits, bool noOutput = false);