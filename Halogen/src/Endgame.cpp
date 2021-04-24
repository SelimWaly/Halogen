#include "Endgame.h"

enum EndGamePatterns
{
	KXvK,
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

int MaterialScore(const Position& position, Players side)
{
	int material = 0;
	material += PieceValues[PAWN] * GetBitCount(position.GetPieceBB(PAWN, side));
	material += PieceValues[KNIGHT] * GetBitCount(position.GetPieceBB(KNIGHT, side));
	material += PieceValues[BISHOP] * GetBitCount(position.GetPieceBB(BISHOP, side));
	material += PieceValues[ROOK] * GetBitCount(position.GetPieceBB(ROOK, side));
	material += PieceValues[QUEEN] * GetBitCount(position.GetPieceBB(QUEEN, side));
	return material;
}

template <EndGamePatterns>
int EndGame(const Position& position, Players stronger) = delete;

template <>
int EndGame<KXvK>(const Position& position, Players stronger)
{
	//Force weaker king to the edges and corners
		//Keep stronger king close to weaker king

	Square strongKing = position.GetKing(stronger);
	Square weakKing = position.GetKing(!stronger);

	int score = MaterialScore(position, stronger)
		+ 20 * CenterDistance[weakKing]
		- 20 * ChebyshevDistance[strongKing][weakKing];

	score = std::min<int>(score + KNOWN_WIN, EVAL_MAX);
	return stronger == WHITE ? score : -score;
}

bool EndGameMatch(const Position& position, int& eval)
{
	Players weaker = N_PLAYERS;

	if (position.GetPiecesColour(BLACK) == position.GetPieceBB(KING, BLACK))
		weaker = BLACK;

	if (position.GetPiecesColour(WHITE) == position.GetPieceBB(KING, WHITE))
		weaker = WHITE;

	if (weaker != N_PLAYERS)
	{
		//KRvK, KQvK with stronger side optionally having additional material
		if (position.GetPieceBB(ROOK, !weaker) || position.GetPieceBB(QUEEN, !weaker))
		{
			eval = EndGame<KXvK>(position, !weaker);
			return true;
		}
	}

	return false;
}