#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "../EGTB.h"
#include "../EvalNet.h"
#include "../GameState.h"
#include "../MoveGeneration.h"
#include "../Search.h"
#include "../SearchData.h"
#include "HalogenNetwork.h"
#include "TrainableNetwork.h"
#include "training_values.h"

void SelfPlayGame(TrainableNetwork& network, SearchSharedState& data, const std::vector<std::string>& openings);
void PrintNetworkDiagnostics(TrainableNetwork& network);
void ExtractGradientsFromTT(
    TrainableNetwork& network, GameState& position, BasicMoveList& line, int distance_from_root);
Score syzygy_rescoring(Score tt_score, BoardState& board);

std::string weight_file_name(int epoch, int game);
std::vector<std::string> parse_opening_book(std::string_view path);

// debug diagnostics
std::atomic<uint64_t> game_count = 0;
std::atomic<uint64_t> white_wins = 0;
std::atomic<uint64_t> draws = 0;
std::atomic<uint64_t> black_wins = 0;
std::atomic<uint64_t> search_count = 0;
std::atomic<uint64_t> total_node_count = 0;

std::atomic<bool> stop_signal = false;

void learn_thread(const std::vector<std::string>& openings)
{
    TrainableNetwork network;
    SearchSharedState data(1);
    data.silent_mode = true;
    data.limits.depth = search_depth;

    while (!stop_signal)
    {
        SelfPlayGame(network, data, openings);
        game_count++;
    }
}

void info_thread(TrainableNetwork& network, int epoch)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_print = start;
    std::chrono::steady_clock::time_point last_save = start;

    while (true)
    {
        // wake up once a second and print diagnostics if needed.
        // by sleeping, we avoid spinning endlessly and contesting shared memory.
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 10)
        {
            auto duration = std::chrono::duration<float>(now - last_print).count();
            lr_alpha = learning_rate_schedule(
                std::chrono::duration<float>(now - start).count() / (60.0 * 60.0 * training_time_hours));

            std::cout << "Games played: " << game_count << std::endl;
            std::cout << "White wins: " << double(white_wins) / double(game_count) << std::endl;
            std::cout << "draws: " << double(draws) / double(game_count) << std::endl;
            std::cout << "Black wins: " << double(black_wins) / double(game_count) << std::endl;
            std::cout << "Searches per second: " << double(search_count) / duration << std::endl;
            std::cout << "Learning rate: " << lr_alpha << std::endl;

            last_print = std::chrono::steady_clock::now();
            search_count = 0;
            total_node_count = 0;

            const auto test_positions = { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
                "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "8/6k1/8/8/8/3R4/2K5/8 w - - 0 1",
                "8/6k1/8/8/8/2BN4/2K5/8 w - - 0 1" };

            GameState test_position;
            for (const auto& fen : test_positions)
            {
                test_position.InitialiseFromFen(fen);
                std::cout << fen << std::endl;
                std::cout << "Eval: " << test_position.GetFloatEvaluation() << std::endl;
            }

            std::cout << std::endl;
            std::cout << std::endl;
        }

        if (std::chrono::duration_cast<std::chrono::minutes>(now - last_save).count() >= 15)
        {
            last_save = std::chrono::steady_clock::now();
            network.SaveWeights(weight_file_name(epoch, game_count));
        }

        if (std::chrono::duration<float>(now - start).count() >= 60.0 * 60.0 * training_time_hours)
        {
            std::cout << "Training complete." << std::endl;
            stop_signal = true;
            network.SaveWeights(weight_file_name(epoch, game_count));
            return;
        }
    }
}

void learn(const std::string& initial_weights_file, int epoch, const std::string& opening_book)
{
    if (!TrainableNetwork::VerifyWeightReadWrite())
    {
        return;
    }

    TrainableNetwork network;

    if (initial_weights_file == "none")
    {
        std::cout << "Initializing weights randomly\n";
        network.InitializeWeightsRandomly();
    }
    else
    {
        std::cout << "Initializing weights from file\n";
        network.LoadWeights(initial_weights_file);
    }

    // Save the initial weights as a baseline
    network.SaveWeights(weight_file_name(epoch, 0));

    // Read in the opening book
    auto openings = parse_opening_book(opening_book);

    std::vector<std::thread> threads;

    threads.emplace_back(info_thread, std::ref(network), epoch);

    for (int i = 0; i < training_threads; i++)
    {
        threads.emplace_back([&]() { learn_thread(openings); });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}

void SelfPlayGame(TrainableNetwork& network, SearchSharedState& data, const std::vector<std::string>& openings)
{
    thread_local std::mt19937 gen(0);

    GameState position;

    const auto& opening = [&openings]() -> std::string
    {
        // pick a random opening, or simply use the starting position
        std::uniform_real_distribution<> opening_prob(0, 1);

        if (opening_prob(gen) < opening_book_usage_pct)
        {
            std::uniform_int_distribution<> opening_line(0, openings.size() - 1);
            return openings[opening_line(gen)];
        }
        else
        {
            // starting position;
            return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
    }();

    bool valid_opening = false;

    while (!valid_opening)
    {
        int turns = 0;
        position.InitialiseFromFen(opening);

        // play out 10 random moves
        while (turns < 10)
        {
            BasicMoveList moves;
            LegalMoves(position.Board(), moves);

            if (moves.size() == 0)
            {
                // checkmate -> reset and generate a new opening line
                break;
            }

            std::uniform_int_distribution<> move(0, moves.size() - 1);
            position.ApplyMove(moves[move(gen)]);
            turns++;
            continue;
        }

        BasicMoveList moves;
        LegalMoves(position.Board(), moves);

        if (moves.size() == 0)
        {
            // checkmate -> reset and generate a new opening line
            continue;
        }

        SearchThread(position, data);
        auto result = data.get_best_search_result();
        auto eval = result.score;

        // check static eval is within set margin
        if (-opening_cutoff < eval && eval < opening_cutoff)
        {
            valid_opening = true;
        }
    }

    GameState game_start = position;
    std::vector<Move> game_moves;
    float game_result = 0;

    while (true)
    {
        // check for a terminal position

        // checkmate or stalemate
        BasicMoveList moves;
        LegalMoves(position.Board(), moves);
        if (moves.size() == 0)
        {
            if (IsInCheck(position.Board()))
            {
                if (position.Board().stm == WHITE)
                {
                    black_wins++;
                    game_result = 0;
                }
                else
                {
                    white_wins++;
                    game_result = 1;
                }
            }
            else
            {
                draws++;
                game_result = 0.5;
            }

            break;
        }

        // 50 move rule
        if (position.Board().fifty_move_count >= 100)
        {
            draws++;
            game_result = 0.5;
            break;
        }

        // 3 fold repitition rule
        if (position.CheckForRep(0, 3))
        {
            draws++;
            game_result = 0.5;
            break;
        }

        // insufficent material rule
        if (DeadPosition(position.Board()))
        {
            draws++;
            game_result = 0.5;
            break;
        }

        // -----------------------------

        SearchThread(position, data);
        BasicMoveList line;

        position.ApplyMove(data.get_best_search_result().best_move);
        game_moves.emplace_back(data.get_best_search_result().best_move);
        search_count++;

        // std::cout << position.Board() << std::endl;
    }

    for (const auto& m : game_moves)
    {
        // We compute the error as the squared difference of the values after a sigmoid is applied. This puts more
        // emphesis on incorrect evals close to even positions, and less on evals where one side is winning.
        //
        // loss = (sigmoid(eval) - sigmoid(tt_score)) ^ 2
        // hence dl/d_eval = 2 * k * s(eval) * (1-s(eval)) * (s(eval) - s(target_score))

        double eval = game_start.GetFloatEvaluation();

        // flip to white POV so the game result makes sense
        eval = game_start.Board().stm == WHITE ? eval : -eval;

        double loss_gradient = 2 * sigmoid_prime(eval) * (sigmoid(eval) - game_result);

        // std::cout << "Eval: " << game_start.GetFloatEvaluation() << " stm: " << game_start.Board().stm
        //           << " result: " << game_result << " loss_gradient: " << loss_gradient << std::endl;
        network.UpdateGradients(loss_gradient, network.GetSparseInputs(game_start.Board()), game_start.Board().stm);
        game_start.ApplyMove(m);
    }

    network.ApplyOptimizationStep(game_moves.size());
}

std::string weight_file_name(int epoch, int game)
{
    // format: 768-512x2-1_r123_g1234567890.nn"
    return "768-" + std::to_string(architecture[1]) + "x2-1_r" + std::to_string(epoch) + "_g" + std::to_string(game)
        + ".nn";
}

std::vector<std::string> parse_opening_book(std::string_view path)
{
    std::ifstream opening_book_file(path.data());

    if (!opening_book_file)
    {
        std::cout << "Error reading opening book file" << std::endl;
        return {};
    }

    std::string line;
    std::vector<std::string> lines;

    while (std::getline(opening_book_file, line))
    {
        lines.emplace_back(std::move(line));
    }

    std::cout << "Read " << lines.size() << " opening lines from file " << path << std::endl;

    return lines;
}
