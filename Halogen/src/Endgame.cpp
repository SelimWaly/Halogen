#include "Endgame.h"

enum EndGamePatterns
{
	KXvK,
	KQvKR,
};

constexpr std::array<int, N_SQUARES> CenterDistance = {
	6, 5, 4, 3, 3, 4, 5, 6,
	5, 4, 3, 2, 2, 3, 4, 5,
	4, 3, 2, 1, 1, 2, 3, 4,
	3, 2, 1, 0, 0, 1, 2, 3,
	3, 2, 1, 0, 0, 1, 2, 3,
	4, 3, 2, 1, 1, 2, 3, 4,
	5, 4, 3, 2, 2, 3, 4, 5,
	6, 5, 4, 3, 3, 4, 5, 6,
};

constexpr auto ChebyshevDistanceInit()
{
	std::array< std::array<int, N_SQUARES>, N_SQUARES> ret = {};

	for (int i = 0; i < N_SQUARES; i++)
		for (int j = 0; j < N_SQUARES; j++)
			ret[i][j] = std::max(AbsFileDiff(i, j), AbsRankDiff(i, j));

	return ret;
}

constexpr auto ChebyshevDistance = ChebyshevDistanceInit();

template <EndGamePatterns, Players stronger>
struct EndGame
{
};

template <Players stronger>
struct EndGame<KXvK, stronger>
{
	int operator()(const Position& position) const
	{
		//Force weaker king to the edges and corners
		//Keep stronger king close to weaker king

		Square strongKing = position.GetKing(stronger);
		Square weakKing = position.GetKing(!stronger);
		
		int score = 100 * CenterDistance[weakKing] - 50 * ChebyshevDistance[strongKing][weakKing] + 200;
		assert(0 <= score && score <= 700);

		score += EVAL_MAX + 1;
		return stronger == WHITE ? score : -score;
	}
};

template <Players stronger>
struct EndGame<KQvKR, stronger>
{
	int operator()(const Position& position) const
	{
		//Force weaker king to the edges and corners
		//Keep stronger king close to weaker king

		Square strongKing = position.GetKing(stronger);
		Square weakKing = position.GetKing(!stronger);

		int score = 100 * CenterDistance[weakKing] - 50 * ChebyshevDistance[strongKing][weakKing] + 200;
		assert(0 <= score && score <= 700);

		return stronger == WHITE ? score : -score;
	}
};

bool EndGameMatch(const Position& position, int& eval)
{
	int count = GetBitCount(position.GetAllPieces());

	if (count == 3)
	{
		//KRvK
		if (position.GetPieceBB(WHITE_ROOK))
		{
			eval = EndGame<KXvK, WHITE>()(position);
			return true;
		}

		if (position.GetPieceBB(BLACK_ROOK))
		{
			eval = EndGame<KXvK, BLACK>()(position);
			return true;
		}

		//KQvK
		if (position.GetPieceBB(WHITE_QUEEN))
		{

			eval = EndGame<KXvK, WHITE>()(position);
			return true;
		}

		if (position.GetPieceBB(BLACK_QUEEN))
		{
			eval = EndGame<KXvK, BLACK>()(position);
			return true;
		}
	}

	if (count == 4)
	{
		//KQvKR
		if (position.GetPieceBB(WHITE_QUEEN) && position.GetPieceBB(BLACK_ROOK))
		{
			eval = EndGame<KQvKR, WHITE>()(position);
			return true;
		}

		//KQvKR
		if (position.GetPieceBB(BLACK_QUEEN) && position.GetPieceBB(WHITE_ROOK))
		{
			eval = EndGame<KQvKR, BLACK>()(position);
			return true;
		}
	}

	return false;
}

