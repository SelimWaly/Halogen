#pragma once

#include "Pyrrhic/tbprobe.h"
#include "MoveGenerator.h"
#include <ctime>
#include <algorithm>
#include <thread>
#include <cmath>

/*Tuneable search constants*/

inline double LMR_constant = -1.38;
inline double LMR_coeff = 1.12;

inline int Null_constant = 5;
inline int Null_depth_quotent = 6;
inline int Null_beta_quotent = 283;

inline int Futility_constant = 69;
inline int Futility_coeff = 48;
inline int Futility_depth = 10;

inline int Aspiration_window = 18;

inline int Delta_margin = 240;

inline int SNMP_coeff = 111;
inline int SNMP_depth = 7;

inline int LMP_constant = 8;
inline int LMP_coeff = 8;
inline int LMP_depth = 6;

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