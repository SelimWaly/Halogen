#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <string>
#include <thread>

#include "../EvalNet.h"
#include "../MoveGeneration.h"
#include "../Position.h"
#include "../Search.h"
#include "../SearchData.h"
#include "HalogenNetwork.h"
#include "TrainableNetwork.h"
#include "td-leaf-learn.h"

void SelfPlayGame(TrainableNetwork& network, ThreadSharedData& data);
void PrintNetworkDiagnostics(TrainableNetwork& network);

// hyperparameters
constexpr double LAMBDA = 0.7; // credit discount factor
constexpr double GAMMA = 1; // discount rate of future rewards

constexpr int training_nodes = 1000;
constexpr double sigmoid_coeff = 2.5 / 400.0;

constexpr double training_time_hours = 24;
// -----------------

constexpr int max_threads = 11;

std::atomic<uint64_t> game_count = 0;

std::atomic<uint64_t> move_count = 0;
std::atomic<uint64_t> depth_count = 0;

std::atomic<bool> stop_signal = false;

void learn_thread()
{
    TrainableNetwork network;
    SearchLimits limits;
    limits.SetNodeLimit(training_nodes);
    ThreadSharedData data(std::move(limits));

    while (!stop_signal)
    {
        SelfPlayGame(network, data);
        game_count++;
    }
}

void info_thread(TrainableNetwork& network)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_print = start;
    std::chrono::steady_clock::time_point last_save = start;
    uint64_t game_count_last = 0;

    while (true)
    {
        // wake up once a second and print diagnostics if needed.
        // by sleeping, we avoid spinning endlessly and contesting shared memory.
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 10)
        {
            auto duration = std::chrono::duration<float>(now - last_print).count();

            std::cout << "Game " << game_count << std::endl;
            std::cout << "Games per second: " << (game_count - game_count_last) / duration << std::endl;
            std::cout << "Average search depth: " << static_cast<double>(depth_count) / static_cast<double>(move_count) << std::endl;
            std::cout << "Learning rate: " << TrainableNetwork::adam_state::alpha << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;

            last_print = std::chrono::steady_clock::now();
            game_count_last = game_count;
            move_count = 0;
            depth_count = 0;

            // cosine annealing
            TrainableNetwork::adam_state::alpha = (cos(std::chrono::duration<float>(now - start).count() * M_PI / (60.0 * 60.0 * training_time_hours)) + 1.0) / 2.0 * 0.001 * 16;
        }

        if (std::chrono::duration_cast<std::chrono::minutes>(now - last_save).count() >= 15)
        {
            last_save = std::chrono::steady_clock::now();
            network.SaveWeights("768-" + std::to_string(architecture[1]) + "x2-1_g" + std::to_string(game_count) + ".nn");
        }

        if (std::chrono::duration<float>(now - start).count() >= 60.0 * 60.0 * training_time_hours)
        {
            std::cout << "Training complete." << std::endl;
            stop_signal = true;
            network.SaveWeights("768-" + std::to_string(architecture[1]) + "x2-1_g" + std::to_string(game_count) + ".nn");
            return;
        }
    }
}

void learn()
{
    TrainableNetwork network;
    network.InitializeWeightsRandomly();

    std::vector<std::thread> threads;

    threads.emplace_back(info_thread, std::ref(network));

    // always have at least one learning and one info thread.
    // at most we want max_threads total threads.
    for (int i = 0; i < std::max(1, max_threads - 1); i++)
    {
        threads.emplace_back(learn_thread);
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(sigmoid_coeff * -x));
}

struct TD_game_result
{
    float score;
    Players stm;
    std::array<std::vector<int>, N_PLAYERS> sparseInputs = {};
    double delta = 0; // between this and next
};

void SelfPlayGame(TrainableNetwork& network, ThreadSharedData& data)
{
    // std::chrono::steady_clock::time_point fn_begin = std::chrono::steady_clock::now();
    // uint64_t time_spend_in_search_ns = 0;

    static std::mt19937 gen(0);
    static std::binomial_distribution<bool> turn(1);

    Position position;
    auto& searchData = data.GetData(0);

    std::vector<TD_game_result> results;

    int turns = 0;
    while (true)
    {
        turns++;

        // check for a terminal position

        // checkmate or stalemate
        BasicMoveList moves;
        LegalMoves(position, moves);
        if (moves.size() == 0)
        {
            if (IsInCheck(position))
            {
                results.push_back({ position.GetTurn() == WHITE ? 0.f : 1.f, position.GetTurn() });
            }
            else
            {
                results.push_back({ 0.5f, position.GetTurn() });
            }

            break;
        }

        // 50 move rule
        if (position.GetFiftyMoveCount() >= 100)
        {
            results.push_back({ 0.5f, position.GetTurn() });
            break;
        }

        // 3 fold repitition rule
        if (position.CheckForRep(0, 3))
        {
            results.push_back({ 0.5f, position.GetTurn() });
            break;
        }

        // insufficent material rule
        if (DeadPosition(position))
        {
            results.push_back({ 0.5f, position.GetTurn() });
            break;
        }

        // apply a random opening if at the start of the game and it's not over yet
        if (turns < 10)
        {
            std::uniform_int_distribution<> move(0, moves.size() - 1);
            position.ApplyMove(moves[move(gen)]);
            continue;
        }

        // -----------------------------

        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        SearchThread(position, data);
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // time_spend_in_search_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        depth_count += data.GetDepth() - 1;
        move_count++;

        const auto& pv = searchData.PvTable[0];

        for (size_t i = 0; i < pv.size(); i++)
        {
            position.ApplyMove(pv[i]);
        }

        /*
        You could arguably use either the eval of the end of the pv or the score returned by the search.
        In theory for basic alpha-beta search these would be equal. That is not the case in practice due to
        search instability (transposition table etc) and search score adjustments (checkmate, draw randomness etc).
        */
        results.push_back({ sigmoid(position.GetEvaluation()), position.GetTurn(), network.GetSparseInputs(position) });

        for (size_t i = 0; i < pv.size(); i++)
        {
            position.RevertMove();
        }

        position.ApplyMove(data.GetBestMove());
    }

    if (results.size() <= 1)
    {
        return;
    }

    for (size_t i = 0; i < results.size() - 1; i++)
    {
        results[i].delta = GAMMA * results[i + 1].score - results[i].score;
        // std::cout << "difference: " << results[i].difference << " scores: " << results[i].score << ", " << results[i + 1].score << std::endl;
    }

    // main td-leaf update step:

    int steps_since_update = 0;

    for (size_t t = 0; t < results.size() - 1; t++)
    {
        steps_since_update++;

        double delta_sum = 0;

        for (size_t j = t; j < results.size() - 1; j++)
        {
            delta_sum += results[j].delta * pow(LAMBDA * GAMMA, j - t);
        }

        // note derivative of sigmoid with coefficent k is k*(s)*(1-s)
        double loss_gradient = -delta_sum * results[t].score * (1 - results[t].score) * sigmoid_coeff;

        // network outputs relative values, but temporal difference is from white's POV
        loss_gradient = results[t].stm == WHITE ? loss_gradient : -loss_gradient;

        network.UpdateGradients(loss_gradient, results[t].sparseInputs, results[t].stm);

        if (steps_since_update >= 16)
        {
            network.ApplyOptimizationStep(steps_since_update);
            steps_since_update = 0;
        }
    }

    if (steps_since_update > 0)
    {
        network.ApplyOptimizationStep(steps_since_update);
    }

    // std::cout << "Game result: " << results.back().score << " turns: " << turns << std::endl;

    // std::chrono::steady_clock::time_point fn_end = std::chrono::steady_clock::now();
    // auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(fn_end - fn_begin).count();
    // std::cout << "Time spend in search: " << static_cast<double>(time_spend_in_search_ns) / static_cast<double>(total_time) * 100 << "%" << std::endl;
}