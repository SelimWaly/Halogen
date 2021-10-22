#include "MoveGeneration.h"

namespace detail
{
	//Top level functions
	template <Players STM, typename T> void GenerateLegalMoves(Position& position, FixedVector<T>& moves, uint64_t pinned);
	template <Players STM, typename T> void AddQuiescenceMoves(Position& position, FixedVector<T>& moves, uint64_t pinned);	//captures and/or promotions
	template <Players STM, typename T> void AddQuietMoves(Position& position, FixedVector<T>& moves, uint64_t pinned);
	template <Players STM, typename T> void InCheckMoves(Position& position, FixedVector<T>& moves, uint64_t pinned);

	//Pawn moves
	template <Players STM, typename T> void PawnPushes(Position& position, FixedVector<T>& moves, uint64_t pinned);
	template <Players STM, typename T> void PawnPromotions(Position& position, FixedVector<T>& moves, uint64_t pinned);
	template <Players STM, typename T> void PawnDoublePushes(Position& position, FixedVector<T>& moves, uint64_t pinned);
	template <Players STM, typename T> void PawnEnPassant(Position& position, FixedVector<T>& moves);	//Ep moves are always checked for legality so no need for pinned mask
	template <Players STM, typename T> void PawnCaptures(Position& position, FixedVector<T>& moves, uint64_t pinned);

	//All other pieces
	template <Players STM, PieceTypes pieceType, bool capture, typename T>
	void GenerateMoves(Position& position, FixedVector<T>& moves, Square square, uint64_t pinned);

	//misc
	template <Players STM, typename T> void CastleMoves(const Position& position, FixedVector<T>& moves);

	//utility functions
	template <Players STM> bool MovePutsSelfInCheck(Position& position, const Move& move);
	template <Players STM> uint64_t PinnedMask(const Position& position);
	bool IsSquareThreatened(const Position & position, Square square, Players colour);		//will tell you if the king WOULD be threatened on that square. Useful for finding defended / threatening pieces
	uint64_t GetThreats(const Position& position, Square square, Players colour);	//colour is of the attacked piece! So to get the black threats of a white piece pass colour = WHITE!

	//special generators for when in check
	template <Players STM, typename T> void KingEvasions(Position& position, FixedVector<T>& moves);						//move the king out of danger	(single or multi threat)
	template <Players STM, typename T> void KingCapturesEvade(Position& position, FixedVector<T>& moves);				//use only for multi threat with king evasions
	template <Players STM, typename T> void CaptureThreat(Position& position, FixedVector<T>& moves, uint64_t threats);	//capture the attacker	(single threat only)
	template <Players STM, typename T> void BlockThreat(Position& position, FixedVector<T>& moves, uint64_t threats);	//block the attacker (single threat only)

	template <Players STM, typename T> inline
	void InCheckMoves(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		uint64_t Threats = GetThreats(position, position.GetKing(STM), STM);
		assert(Threats != 0);

		if (GetBitCount(Threats) > 1)					//double check
		{
			KingEvasions<STM>(position, moves);
			KingCapturesEvade<STM>(position, moves);
		}
		else
		{
			PawnPushes<STM>(position, moves, pinned);		//pawn moves are hard :( so we calculate those normally
			PawnDoublePushes<STM>(position, moves, pinned);
			PawnCaptures<STM>(position, moves, pinned);
			PawnEnPassant<STM>(position, moves);
			PawnPromotions<STM>(position, moves, pinned);

			KingEvasions<STM>(position, moves);
			KingCapturesEvade<STM>(position, moves);
			CaptureThreat<STM>(position, moves, Threats);
			BlockThreat<STM>(position, moves, Threats);
		}
	}

	template <Players STM, typename T> inline
	void AddQuiescenceMoves(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		PawnCaptures<STM>(position, moves, pinned);
		PawnEnPassant<STM>(position, moves);
		PawnPromotions<STM>(position, moves, pinned);

		for (uint64_t pieces = position.GetPieceBB(KNIGHT, STM); pieces != 0; GenerateMoves<STM, KNIGHT, true>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(BISHOP, STM); pieces != 0; GenerateMoves<STM, BISHOP, true>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(KING, STM); pieces != 0; GenerateMoves<STM, KING, true>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(ROOK, STM); pieces != 0; GenerateMoves<STM, ROOK, true>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(QUEEN, STM); pieces != 0; GenerateMoves<STM, QUEEN, true>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
	}

	template <Players STM>
	inline uint64_t PinnedMask(const Position& position)
	{
		Square king = position.GetKing(STM);
		if (IsInCheck(position)) return UNIVERSE;

		uint64_t mask = EMPTY;
		uint64_t possiblePins = QueenAttacks[king] & position.GetPiecesColour(STM);
		uint64_t maskAll = position.GetAllPieces();

		while (possiblePins != 0)
		{
			unsigned int sq = LSBpop(possiblePins);

			if (!mayMove(king, sq, maskAll))	//if you can't move from the square to the king, it can't be pinned
				continue;

			//If a piece is moving from the same diagonal as the king, and that diagonal contains an enemy bishop or queen
			if ((GetDiagonal(king) == GetDiagonal(sq)) && (DiagonalBB[GetDiagonal(king)] & (position.GetPieceBB(BISHOP, !STM) | position.GetPieceBB(QUEEN, !STM))))
			{
				mask |= SquareBB[sq];
				continue;
			}

			//If a piece is moving from the same anti-diagonal as the king, and that diagonal contains an enemy bishop or queen
			if ((GetAntiDiagonal(king) == GetAntiDiagonal(sq)) && (AntiDiagonalBB[GetAntiDiagonal(king)] & (position.GetPieceBB(BISHOP, !STM) | position.GetPieceBB(QUEEN, !STM))))
			{
				mask |= SquareBB[sq];
				continue;
			}

			//If a piece is moving from the same file as the king, and that file contains an enemy rook or queen
			if ((GetFile(king) == GetFile(sq)) && (FileBB[GetFile(king)] & (position.GetPieceBB(ROOK, !STM) | position.GetPieceBB(QUEEN, !STM))))
			{
				mask |= SquareBB[sq];
				continue;
			}

			//If a piece is moving from the same rank as the king, and that rank contains an enemy rook or queen
			if ((GetRank(king) == GetRank(sq)) && (RankBB[GetRank(king)] & (position.GetPieceBB(ROOK, !STM) | position.GetPieceBB(QUEEN, !STM))))
			{
				mask |= SquareBB[sq];
				continue;
			}
		}

		mask |= SquareBB[king];
		return mask;
	}

	//Moves going from a square to squares on a bitboard
	template <Players STM, typename T> inline
	void AppendLegalMoves(Square from, uint64_t to, Position& position, MoveFlag flag, FixedVector<T>& moves, uint64_t pinned = UNIVERSE)
	{
		while (to != 0)
		{
			Square target = static_cast<Square>(LSBpop(to));
			Move move(from, target, flag);
			if (!(pinned & SquareBB[from]) || !MovePutsSelfInCheck<STM>(position, move))
				moves.Append(move);
		}
	}

	//Moves going to a square from squares on a bitboard
	template <Players STM, typename T> inline
	void AppendLegalMoves(uint64_t from, Square to, Position& position, MoveFlag flag, FixedVector<T>& moves, uint64_t pinned = UNIVERSE)
	{
		while (from != 0)
		{
			Square source = static_cast<Square>(LSBpop(from));
			Move move(source, to, flag);
			if (!(pinned & SquareBB[source]) || !MovePutsSelfInCheck<STM>(position, move))
				moves.Append(move);
		}
	}

	template <Players STM, typename T> inline
	void KingEvasions(Position& position, FixedVector<T>& moves)
	{
		Square square = position.GetKing(STM);
		uint64_t quiets = position.GetEmptySquares() & KingAttacks[square];
		AppendLegalMoves<STM>(square, quiets, position, QUIET, moves);
	}

	template <Players STM, typename T> inline
	void KingCapturesEvade(Position& position, FixedVector<T>& moves)
	{
		Square square = position.GetKing(STM);
		uint64_t captures = (position.GetPiecesColour(!STM)) & KingAttacks[square];
		AppendLegalMoves<STM>(square, captures, position, CAPTURE, moves);
	}

	template <Players STM, typename T> inline
	void CaptureThreat(Position& position, FixedVector<T>& moves, uint64_t threats)
	{
		Square square = static_cast<Square>(LSBpop(threats));

		uint64_t potentialCaptures = GetThreats(position, square, !STM)
			& ~SquareBB[position.GetKing(STM)]									//King captures handelled in KingCapturesEvade()
			& ~position.GetPieceBB(Piece(PAWN, STM));							//Pawn captures handelled elsewhere

		AppendLegalMoves<STM>(potentialCaptures, square, position, CAPTURE, moves);
	}

	template <Players STM, typename T> inline
	void BlockThreat(Position& position, FixedVector<T>& moves, uint64_t threats)
	{
		Square threatSquare = static_cast<Square>(LSBpop(threats));
		Pieces piece = position.GetSquare(threatSquare);

		if (piece == WHITE_PAWN || piece == BLACK_PAWN || piece == WHITE_KNIGHT || piece == BLACK_KNIGHT) return;	//cant block non-sliders. Also cant be threatened by enemy king

		uint64_t blockSquares = betweenArray[threatSquare][position.GetKing(STM)];

		while (blockSquares != 0)
		{
			Square square = static_cast<Square>(LSBpop(blockSquares));
			uint64_t potentialBlockers = GetThreats(position, square, !STM) & ~position.GetPieceBB(Piece(PAWN, STM));	//pawn moves need to be handelled elsewhere because they might threaten a square without being able to move there
			AppendLegalMoves<STM>(potentialBlockers, square, position, QUIET, moves);
		}
	}

	template <Players STM, typename T> inline
	void GenerateLegalMoves(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		AddQuietMoves<STM>(position, moves, pinned);
		AddQuiescenceMoves<STM>(position, moves, pinned);
	}

	template <Players STM, typename T> inline
	void AddQuietMoves(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		PawnPushes<STM>(position, moves, pinned);
		PawnDoublePushes<STM>(position, moves, pinned);
		CastleMoves<STM>(position, moves);

		for (uint64_t pieces = position.GetPieceBB(KNIGHT, STM); pieces != 0; GenerateMoves<STM, KNIGHT, false>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(BISHOP, STM); pieces != 0; GenerateMoves<STM, BISHOP, false>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(QUEEN,  STM); pieces != 0; GenerateMoves<STM, QUEEN,  false>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(ROOK,   STM); pieces != 0; GenerateMoves<STM, ROOK,   false>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
		for (uint64_t pieces = position.GetPieceBB(KING,   STM); pieces != 0; GenerateMoves<STM, KING,   false>(position, moves, static_cast<Square>(LSBpop(pieces)), pinned));
	}

	template <Players STM, typename T> inline
	void PawnPushes(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		int foward = 0;
		uint64_t targets = 0;
		uint64_t pawnSquares = position.GetPieceBB(PAWN, STM);

		if (STM == WHITE)
		{
			foward = 8;
			targets = (pawnSquares << 8) & position.GetEmptySquares();
		}
		if (STM == BLACK)
		{
			foward = -8;
			targets = (pawnSquares >> 8) & position.GetEmptySquares();
		}

		uint64_t pawnPushes = targets & ~(RankBB[RANK_1] | RankBB[RANK_8]);			//pushes that aren't to the back ranks

		while (pawnPushes != 0)
		{
			Square end = static_cast<Square>(LSBpop(pawnPushes));
			Square start = static_cast<Square>(end - foward);

			Move move(start, end, QUIET);

			if (!(pinned & SquareBB[start]) || !MovePutsSelfInCheck<STM>(position, move))
				moves.Append(move);
		}
	}

	template <Players STM, typename T> inline
	void PawnPromotions(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		int foward = 0;
		uint64_t targets = 0;
		uint64_t pawnSquares = position.GetPieceBB(PAWN, STM);

		if (STM == WHITE)
		{
			foward = 8;
			targets = (pawnSquares << 8) & position.GetEmptySquares();
		}
		if (STM == BLACK)
		{
			foward = -8;
			targets = (pawnSquares >> 8) & position.GetEmptySquares();
		}

		uint64_t pawnPromotions = targets & (RankBB[RANK_1] | RankBB[RANK_8]);			//pushes that are to the back ranks

		while (pawnPromotions != 0)
		{
			Square end = static_cast<Square>(LSBpop(pawnPromotions));
			Square start = static_cast<Square>(end - foward);

			Move move(start, end, KNIGHT_PROMOTION);
			if ((pinned & SquareBB[start]) && MovePutsSelfInCheck<STM>(position, move))
				continue;

			moves.Append(move);
			moves.Append(start, end, ROOK_PROMOTION);
			moves.Append(start, end, BISHOP_PROMOTION);
			moves.Append(start, end, QUEEN_PROMOTION);
		}
	}

	template <Players STM, typename T> inline
	void PawnDoublePushes(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		int foward = 0;
		uint64_t targets = 0;
		uint64_t pawnSquares = position.GetPieceBB(PAWN, STM);

		if (STM == WHITE)
		{
			foward = 16;
			pawnSquares &= RankBB[RANK_2];
			targets = (pawnSquares << 8) & position.GetEmptySquares();
			targets = (targets << 8) & position.GetEmptySquares();
		}
		if (STM == BLACK)
		{
			foward = -16;
			pawnSquares &= RankBB[RANK_7];
			targets = (pawnSquares >> 8) & position.GetEmptySquares();
			targets = (targets >> 8) & position.GetEmptySquares();
		}

		while (targets != 0)
		{
			Square end = static_cast<Square>(LSBpop(targets));
			Square start = static_cast<Square>(end - foward);

			Move move(start, end, PAWN_DOUBLE_MOVE);

			if (!(pinned & SquareBB[start]) || !MovePutsSelfInCheck<STM>(position, move))
				moves.Append(move);
		}
	}

	template <Players STM, typename T> inline
	void PawnEnPassant(Position& position, FixedVector<T>& moves)
	{
		if (position.GetEnPassant() <= SQ_H8)
		{
			uint64_t potentialAttackers = PawnAttacks[!STM][position.GetEnPassant()] & position.GetPieceBB(Piece(PAWN, STM));
			AppendLegalMoves<STM>(potentialAttackers, position.GetEnPassant(), position, EN_PASSANT, moves);
		}
	}

	template <Players STM, typename T> inline
	void PawnCaptures(Position& position, FixedVector<T>& moves, uint64_t pinned)
	{
		int fowardleft = 0;
		int fowardright = 0;
		uint64_t leftAttack = 0;
		uint64_t rightAttack = 0;
		uint64_t pawnSquares = position.GetPieceBB(PAWN, STM);

		if (STM == WHITE)
		{
			fowardleft = 7;
			fowardright = 9;
			leftAttack = ((pawnSquares & ~(FileBB[FILE_A])) << 7) & position.GetBlackPieces();
			rightAttack = ((pawnSquares & ~(FileBB[FILE_H])) << 9) & position.GetBlackPieces();
		}
		if (STM == BLACK)
		{
			fowardleft = -9;
			fowardright = -7;
			leftAttack = ((pawnSquares & ~(FileBB[FILE_A])) >> 9) & position.GetWhitePieces();
			rightAttack = ((pawnSquares & ~(FileBB[FILE_H])) >> 7) & position.GetWhitePieces();
		}

		while (leftAttack != 0)
		{
			Square end = static_cast<Square>(LSBpop(leftAttack));
			Square start = static_cast<Square>(end - fowardleft);

			Move move(start, end, CAPTURE);
			if ((pinned & SquareBB[start]) && MovePutsSelfInCheck<STM>(position, move))
				continue;

			if (GetRank(end) == RANK_1 || GetRank(end) == RANK_8)
			{
				moves.Append(start, end, KNIGHT_PROMOTION_CAPTURE);
				moves.Append(start, end, ROOK_PROMOTION_CAPTURE);
				moves.Append(start, end, BISHOP_PROMOTION_CAPTURE);
				moves.Append(start, end, QUEEN_PROMOTION_CAPTURE);
			}
			else
				moves.Append(move);
		}

		while (rightAttack != 0)
		{
			Square end = static_cast<Square>(LSBpop(rightAttack));
			Square start = static_cast<Square>(end - fowardright);

			Move move(start, end, CAPTURE);
			if ((pinned & SquareBB[start]) && MovePutsSelfInCheck<STM>(position, move))
				continue;

			if (GetRank(end) == RANK_1 || GetRank(end) == RANK_8)
			{
				moves.Append(start, end, KNIGHT_PROMOTION_CAPTURE);
				moves.Append(start, end, ROOK_PROMOTION_CAPTURE);
				moves.Append(start, end, BISHOP_PROMOTION_CAPTURE);
				moves.Append(start, end, QUEEN_PROMOTION_CAPTURE);
			}
			else
				moves.Append(move);
		}
	}

	template <Players STM>
	inline void CastleMoves(const Position& position, std::vector<Move>& moves)
	{
		uint64_t Pieces = position.GetAllPieces();

		if (position.GetCanCastleWhiteKingside() && STM == WHITE)
		{
			if (mayMove(SQ_E1, SQ_H1, Pieces))
			{
				if (!IsSquareThreatened(position, SQ_E1, STM) && !IsSquareThreatened(position, SQ_F1, STM) && !IsSquareThreatened(position, SQ_G1, STM))
				{
					moves.emplace_back(SQ_E1, SQ_G1, KING_CASTLE);
				}
			}
		}

		if (position.GetCanCastleWhiteQueenside() && STM == WHITE)
		{
			if (mayMove(SQ_E1, SQ_A1, Pieces))
			{
				if (!IsSquareThreatened(position, SQ_E1, STM) && !IsSquareThreatened(position, SQ_D1, STM) && !IsSquareThreatened(position, SQ_C1, STM))
				{
					moves.emplace_back(SQ_E1, SQ_C1, QUEEN_CASTLE);
				}
			}
		}

		if (position.GetCanCastleBlackKingside() && STM == BLACK)
		{
			if (mayMove(SQ_E8, SQ_H8, Pieces))
			{
				if (!IsSquareThreatened(position, SQ_E8, STM) && !IsSquareThreatened(position, SQ_F8, STM) && !IsSquareThreatened(position, SQ_G8, STM))
				{
					moves.emplace_back(SQ_E8, SQ_G8, KING_CASTLE);
				}
			}
		}

		if (position.GetCanCastleBlackQueenside() && STM == BLACK)
		{
			if (mayMove(SQ_E8, SQ_A8, Pieces))
			{
				if (!IsSquareThreatened(position, SQ_E8, STM) && !IsSquareThreatened(position, SQ_D8, STM) && !IsSquareThreatened(position, SQ_C8, STM))
				{
					moves.emplace_back(SQ_E8, SQ_C8, QUEEN_CASTLE);
				}
			}
		}
	}

	template <Players STM, typename T> inline
	void CastleMoves(const Position& position, FixedVector<T>& moves)
	{
		std::vector<Move> tmp;
		CastleMoves<STM>(position, tmp);
		for (auto& move : tmp)
			moves.Append(move);
	}

	template <Players STM, PieceTypes pieceType, bool capture, typename T> inline
	void GenerateMoves(Position& position, FixedVector<T>& moves, Square square, uint64_t pinned)
	{
		uint64_t occupied = position.GetAllPieces();
		uint64_t targets = (capture ? position.GetPiecesColour(!STM) : ~occupied) & AttackBB<pieceType>(square, occupied);
		AppendLegalMoves<STM>(square, targets, position, capture ? CAPTURE : QUIET, moves, pinned);
	}

	inline bool IsSquareThreatened(const Position& position, Square square, Players colour)
	{
		if ((KnightAttacks[square] & position.GetPieceBB(KNIGHT, !colour)) != 0)
			return true;

		if ((PawnAttacks[colour][square] & position.GetPieceBB(Piece(PAWN, !colour))) != 0)
			return true;

		if ((KingAttacks[square] & position.GetPieceBB(KING, !colour)) != 0)					//if I can attack the enemy king he can attack me
			return true;

		uint64_t Pieces = position.GetAllPieces();
		
		uint64_t queen = position.GetPieceBB(QUEEN, !colour) & QueenAttacks[square];
		while (queen != 0)
		{
			unsigned int start = LSBpop(queen);
			if (mayMove(start, square, Pieces))
				return true;
		}

		uint64_t bishops = position.GetPieceBB(BISHOP, !colour) & BishopAttacks[square];
		while (bishops != 0)
		{
			unsigned int start = LSBpop(bishops);
			if (mayMove(start, square, Pieces))
				return true;
		}

		uint64_t rook = position.GetPieceBB(ROOK, !colour) & RookAttacks[square];
		while (rook != 0)
		{
			unsigned int start = LSBpop(rook);
			if (mayMove(start, square, Pieces))
				return true;
		}

		return false;
	}

	inline uint64_t GetThreats(const Position& position, Square square, Players colour)
	{
		uint64_t threats = EMPTY;

		threats |= (KnightAttacks[square] & position.GetPieceBB(KNIGHT, !colour));
		threats |= (PawnAttacks[colour][square] & position.GetPieceBB(Piece(PAWN, !colour)));
		threats |= (KingAttacks[square] & position.GetPieceBB(KING, !colour));

		uint64_t Pieces = position.GetAllPieces();

		uint64_t queen = position.GetPieceBB(QUEEN, !colour) & QueenAttacks[square];
		while (queen != 0)
		{
			unsigned int start = LSBpop(queen);
			if (mayMove(start, square, Pieces))
				threats |= SquareBB[start];
		}

		uint64_t bishops = position.GetPieceBB(BISHOP, !colour) & BishopAttacks[square];
		while (bishops != 0)
		{
			unsigned int start = LSBpop(bishops);
			if (mayMove(start, square, Pieces))
				threats |= SquareBB[start];
		}

		uint64_t rook = position.GetPieceBB(ROOK, !colour) & RookAttacks[square];
		while (rook != 0)
		{
			unsigned int start = LSBpop(rook);
			if (mayMove(start, square, Pieces))
				threats |= SquareBB[start];
		}

		return threats;
	}

	/*
	This function does not work for casteling moves. They are tested for legality their creation.
	*/
	template <Players STM> inline
	bool MovePutsSelfInCheck(Position& position, const Move & move)
	{
		assert(move.GetFlag() != KING_CASTLE);
		assert(move.GetFlag() != QUEEN_CASTLE);

		position.ApplyMoveQuick(move);
		bool ret = IsInCheck(position, STM);
		position.RevertMoveQuick();
		return ret;
	}
}

inline bool IsInCheck(const Position& position, Players colour)
{
	return detail::IsSquareThreatened(position, position.GetKing(colour), colour);
}

inline bool IsInCheck(const Position& position)
{
	return IsInCheck(position, position.GetTurn());
}

template <typename T> inline
void LegalMoves(Position& position, FixedVector<T>& moves)
{
	if (position.GetTurn() == WHITE)
	{
		if (IsInCheck(position))
		{
			detail::InCheckMoves<WHITE>(position, moves, detail::PinnedMask<WHITE>(position));
		}
		else
		{
			detail::GenerateLegalMoves<WHITE>(position, moves, detail::PinnedMask<WHITE>(position));
		} 
	}
	else
	{
		if (IsInCheck(position))
		{
			detail::InCheckMoves<BLACK>(position, moves, detail::PinnedMask<BLACK>(position));
		}
		else
		{
			detail::GenerateLegalMoves<BLACK>(position, moves, detail::PinnedMask<BLACK>(position));
		} 
	}
}

template <typename T> inline
void QuiescenceMoves(Position& position, FixedVector<T>& moves)
{
	if (position.GetTurn() == WHITE)
	{
		detail::AddQuiescenceMoves<WHITE>(position, moves, detail::PinnedMask<WHITE>(position));
	}
	else
	{
		detail::AddQuiescenceMoves<BLACK>(position, moves, detail::PinnedMask<BLACK>(position));
	}
}

template <typename T> inline
void QuietMoves(Position& position, FixedVector<T>& moves)
{
	if (position.GetTurn() == WHITE)
	{
		detail::AddQuietMoves<WHITE>(position, moves, detail::PinnedMask<WHITE>(position));
	}
	else
	{
		detail::AddQuietMoves<BLACK>(position, moves, detail::PinnedMask<BLACK>(position));
	}
}

inline bool MoveIsLegal(Position& position, const Move& move)
{
	/*Obvious check first*/
	if (move == Move::Uninitialized)
		return false;

	Pieces piece = position.GetSquare(move.GetFrom());

	/*Make sure there's a piece to be moved*/
	if (position.GetSquare(move.GetFrom()) == N_PIECES)
		return false;

	/*Make sure the piece are are moving is ours*/
	if (ColourOfPiece(piece) != position.GetTurn())
		return false;

	/*Make sure we aren't capturing our own piece*/
	if (position.GetSquare(move.GetTo()) != N_PIECES && ColourOfPiece(position.GetSquare(move.GetTo())) == position.GetTurn())
		return false;

	/*We don't use these flags*/
	if (move.GetFlag() == DONT_USE_1 || move.GetFlag() == DONT_USE_2)
		return false;

	uint64_t allPieces = position.GetAllPieces();

	/*Anything in the way of sliding pieces?*/
	if (piece == WHITE_BISHOP || piece == BLACK_BISHOP || piece == WHITE_ROOK || piece == BLACK_ROOK || piece == WHITE_QUEEN || piece == BLACK_QUEEN)
	{
		if (!mayMove(move.GetFrom(), move.GetTo(), allPieces))
			return false;
	}

	/*Pawn moves are complex*/
	if (GetPieceType(piece) == PAWN)
	{
		int forward = piece == WHITE_PAWN ? 1 : -1;
		Rank startingRank = piece == WHITE_PAWN ? RANK_2 : RANK_7;

		//pawn push
		if (RankDiff(move.GetTo(), move.GetFrom()) == forward && FileDiff(move.GetFrom(), move.GetTo()) == 0)
		{
			if (position.IsOccupied(move.GetTo()))			//Something in the way!
				return false;
		}

		//pawn double push
		else if (RankDiff(move.GetTo(), move.GetFrom()) == forward * 2 && FileDiff(move.GetFrom(), move.GetTo()) == 0)
		{
			if (GetRank(move.GetFrom()) != startingRank)	//double move not from starting rank
				return false;
			if (position.IsOccupied(move.GetTo()))			//something on target square
				return false;
			if (!position.IsEmpty(static_cast<Square>((move.GetTo() + move.GetFrom()) / 2)))	//something in between
				return false;
		}

		//pawn capture (not EP)
		else if (RankDiff(move.GetTo(), move.GetFrom()) == forward && AbsFileDiff(move.GetFrom(), move.GetTo()) == 1 && position.GetEnPassant() != move.GetTo())	
		{
			if (position.IsEmpty(move.GetTo()))				//nothing there to capture
				return false;
		}

		//pawn capture (EP)
		else if (RankDiff(move.GetTo(), move.GetFrom()) == forward && AbsFileDiff(move.GetFrom(), move.GetTo()) == 1 && position.GetEnPassant() == move.GetTo())
		{
			if (position.IsEmpty(GetPosition(GetFile(move.GetTo()), GetRank(move.GetFrom()))))	//nothing there to capture
				return false;
		}

		else
		{
			return false;
		}
	}

	/*Check the pieces can actually move as required*/
	if (piece == WHITE_KNIGHT || piece == BLACK_KNIGHT)
	{
		if ((SquareBB[move.GetTo()] & KnightAttacks[move.GetFrom()]) == 0)
			return false;
	}

	if ((piece == WHITE_KING || piece == BLACK_KING) && !(move.GetFlag() == KING_CASTLE || move.GetFlag() == QUEEN_CASTLE))
	{
		if ((SquareBB[move.GetTo()] & KingAttacks[move.GetFrom()]) == 0)
			return false;
	}

	if (piece == WHITE_ROOK || piece == BLACK_ROOK)
	{
		if ((SquareBB[move.GetTo()] & RookAttacks[move.GetFrom()]) == 0)
			return false;
	}

	if (piece == WHITE_BISHOP || piece == BLACK_BISHOP)
	{
		if ((SquareBB[move.GetTo()] & BishopAttacks[move.GetFrom()]) == 0)
			return false;
	}

	if (piece == WHITE_QUEEN || piece == BLACK_QUEEN)
	{
		if ((SquareBB[move.GetTo()] & QueenAttacks[move.GetFrom()]) == 0)
			return false;
	}

	/*For castle moves, just generate them and see if we find a match*/
	if (move.GetFlag() == KING_CASTLE || move.GetFlag() == QUEEN_CASTLE)
	{
		std::vector<Move> moves;

		if (position.GetTurn() == WHITE)
			detail::CastleMoves<WHITE>(position, moves);
		else
			detail::CastleMoves<BLACK>(position, moves);

		for (size_t i = 0; i < moves.size(); i++)
		{
			if (moves[i] == move)
				return true;
		}
		return false;
	}

	//-----------------------------
	//Now, we make sure the moves flag makes sense given the move

	MoveFlag flag = QUIET;

	//Captures
	if (position.IsOccupied(move.GetTo()))
		flag = CAPTURE;

	//Double pawn moves
	if (AbsRankDiff(move.GetFrom(), move.GetTo()) == 2)
		if (position.GetSquare(move.GetFrom()) == WHITE_PAWN || position.GetSquare(move.GetFrom()) == BLACK_PAWN)
			flag = PAWN_DOUBLE_MOVE;

	//En passant
	if (move.GetTo() == position.GetEnPassant())
		if (position.GetSquare(move.GetFrom()) == WHITE_PAWN || position.GetSquare(move.GetFrom()) == BLACK_PAWN)
			flag = EN_PASSANT;

	//Castling
	if (move.GetFrom() == SQ_E1 && move.GetTo() == SQ_G1 && position.GetSquare(move.GetFrom()) == WHITE_KING)
		flag = KING_CASTLE;

	if (move.GetFrom() == SQ_E1 && move.GetTo() == SQ_C1 && position.GetSquare(move.GetFrom()) == WHITE_KING)
		flag = QUEEN_CASTLE;

	if (move.GetFrom() == SQ_E8 && move.GetTo() == SQ_G8 && position.GetSquare(move.GetFrom()) == BLACK_KING)
		flag = KING_CASTLE;

	if (move.GetFrom() == SQ_E8 && move.GetTo() == SQ_C8 && position.GetSquare(move.GetFrom()) == BLACK_KING)
		flag = QUEEN_CASTLE;

	//Promotion
	if (   (position.GetSquare(move.GetFrom()) == WHITE_PAWN && GetRank(move.GetTo()) == RANK_8) 
		|| (position.GetSquare(move.GetFrom()) == BLACK_PAWN && GetRank(move.GetTo()) == RANK_1))
	{
		if (position.IsOccupied(move.GetTo()))
		{
			if (move.GetFlag() != KNIGHT_PROMOTION_CAPTURE
			 && move.GetFlag() != ROOK_PROMOTION_CAPTURE
			 && move.GetFlag() != QUEEN_PROMOTION_CAPTURE
			 && move.GetFlag() != BISHOP_PROMOTION_CAPTURE)
				return false;
		}
		else
		{
			if (move.GetFlag() != KNIGHT_PROMOTION
			 && move.GetFlag() != ROOK_PROMOTION
			 && move.GetFlag() != QUEEN_PROMOTION
			 && move.GetFlag() != BISHOP_PROMOTION)
				return false;
		}
	}
	else
	{
		//check the decided on flag matches
		if (flag != move.GetFlag())
			return false;
	}
	//-----------------------------

	/*Move puts me in check*/
	if (position.GetTurn() == WHITE)
	{
		if (detail::MovePutsSelfInCheck<WHITE>(position, move))
			return false;
	}
	else
	{
		if (detail::MovePutsSelfInCheck<BLACK>(position, move))
			return false;
	}

	return true;
}

//--------------------------------------------------------------------------
// Thanks Terje
// Returns the attack bitboard for a piece of piecetype on square sq
template <PieceTypes pieceType> inline
uint64_t AttackBB(Square sq, uint64_t occupied)
{
	switch (pieceType)
	{
	case KNIGHT:	return KnightAttacks[sq];
	case KING:		return KingAttacks[sq];
	case BISHOP:	return bishopTable.AttackMask(sq, occupied);
	case ROOK:		return rookTable.AttackMask(sq, occupied);
	case QUEEN:		return AttackBB<ROOK>(sq, occupied) | AttackBB<BISHOP>(sq, occupied);
	default:		throw std::invalid_argument("piecetype is argument is invalid");
	}
}
//--------------------------------------------------------------------------