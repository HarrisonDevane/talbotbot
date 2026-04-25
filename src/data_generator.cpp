#define NOMINMAX
#include "data_generator.hpp"
#include "logger.hpp"
#include <iostream>
#include <cstring>
#include <c10/util/Half.h>
#include "board_utils.hpp"
#include <sstream>
#include <windows.h> 
#include <cmath>

DataGenerator::DataGenerator(
    const YAML::Node& global_cfg,
    const YAML::Node& data_gen_cfg, const YAML::Node& mcts_cfg, const YAML::Node& sel_cfg, const YAML::Node& model_cfg,
    const std::string& rl_dir_in, int rot_interval,
    Logger& logger,
    moodycamel::ConcurrentQueue<std::pair<int, int>>& i_queue,
    std::vector<ThreadSafeQueue<std::vector<int>>>& r_queues,
    std::vector<torch::Tensor>& in_buffer, std::vector<torch::Tensor>& p_buffer, std::vector<torch::Tensor>& v_buffer,
    ThreadSafeQueue<int>& free_slots, ThreadSafeQueue<CompletedGame>& completed_games_queue,
    int start_game_id, std::atomic<uint64_t>& step_ref
) : main_logger(logger), inference_queue(i_queue), result_queues(r_queues),
    shared_input_buffer(in_buffer), shared_policy_buffer(p_buffer), shared_value_buffer(v_buffer),
    buffer_free_slots(free_slots), completed_games_queue(completed_games_queue),
    stop_event(false), game_counter(start_game_id), interval_games(0), interval_samples(0), current_step(step_ref)
{
    config.num_cores = data_gen_cfg["game_worker_cores"].size();
    config.workers_per_core = data_gen_cfg["workers_per_core"].as<int>();
    config.total_workers = config.num_cores * config.workers_per_core;
    for (const auto& core : data_gen_cfg["game_worker_cores"]) {
        config.core_ids.push_back(core.as<int>());
    }
    config.max_ply_length = data_gen_cfg["max_ply_length"].as<int>();
    config.worker_logging_level = data_gen_cfg["worker_logging_level"].as<int>();
    config.rl_dir = rl_dir_in;
    config.rotation_interval = rot_interval;

    for (int i = 0; i < config.num_cores; ++i) {
        core_wait_counts.push_back(std::make_unique<std::atomic<int>>(0));
    }

    model_config.input_planes = model_cfg["model"]["input_planes"].as<int>();
    model_config.board_dim = model_cfg["chess"]["board_dim"].as<int>();
    model_config.policy_moves = model_cfg["chess"]["total_policy_moves"].as<int>();

    selector_config.node_pool_size = mcts_cfg["node_pool_size"].as<int>();
    selector_config.batch_size_per_worker = mcts_cfg["worker_minibatch_size"].as<int>();
    selector_config.virtual_loss = mcts_cfg["virtual_loss"].as<double>();
    selector_config.gumbel_c_visit = mcts_cfg["gumbel_c_visit"].as<double>();
    selector_config.gumbel_c_scale = mcts_cfg["gumbel_c_scale"].as<double>();
    selector_config.gumbel_noise = mcts_cfg["gumbel_noise"].as<double>();
    selector_config.gumbel_search_depth = mcts_cfg["gumbel_search_depth"].as<int>();
    selector_config.gumbel_m = mcts_cfg["gumbel_m"].as<int>();
    selector_config.minimax_target_override = mcts_cfg["minimax_target_override"].as<bool>();
    selector_config.minimax_win_target = mcts_cfg["minimax_win_target"].as<double>();
    selector_config.minimax_loss_target = mcts_cfg["minimax_loss_target"].as<double>();
    selector_config.temperature_ply_cutoff = sel_cfg["temperature_ply_cutoff"].as<int>();
    selector_config.temperature_q_decay = sel_cfg["temperature_q_decay"].as<double>();
    selector_config.draw_cutoff = sel_cfg["draw_cutoff"].as<double>();
    selector_config.resignation_probability = sel_cfg["resignation_probability"].as<double>();
    selector_config.resignation_cutoff = sel_cfg["resignation_cutoff"].as<double>();

    main_logger.log("INFO", "DataGenerator logic loop initialized.");
}

// [Destructor, start, stop, _generate_pgn identical to current...]
DataGenerator::~DataGenerator() { stop(); }
void DataGenerator::start() {
    int logical_idx = 0;
    for (int i = 0; i < config.num_cores; ++i) {
        for (int w = 0; w < config.workers_per_core; ++w) {
            workers.emplace_back(&DataGenerator::worker_main, this, logical_idx, config.core_ids[i]);
            logical_idx++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
        }
    }
}
void DataGenerator::stop() {
    stop_event.store(true);
    for (auto& t : workers) if (t.joinable()) t.join();
}

void DataGenerator::_generate_pgn(int game_number, const std::vector<GameTransition>& transitions, const std::string& result_str, Logger& logger) {
    std::stringstream pgn;
    pgn << "[Event \"Self-Play Game " << game_number << "\"]\n";
    pgn << "[Site \"Talbot C++ Engine\"]\n";
    pgn << "[Result \"" << result_str << "\"]\n";
    pgn << "[White \"Talbot Agent\"]\n";
    pgn << "[Black \"Talbot Agent\"]\n\n";

    chess::Board temp_board;
    temp_board.setFen(chess::constants::STARTPOS);

    for (size_t i = 0; i < transitions.size(); ++i) {
        if (i % 12 == 0 && i != 0) pgn << "\n";
        if (i % 2 == 0) pgn << (i / 2 + 1) << ". ";
        pgn << chess::uci::moveToSan(temp_board, transitions[i].move) << " ";
        temp_board.makeMove(transitions[i].move);
    }
    
    pgn << result_str << "\n";
    logger.log("CRITICAL", "Game PGN:\n" + pgn.str());
}

void DataGenerator::worker_main(int logical_idx, int core_id) {
    int worker_id = logical_idx + 1;
    int core_index = logical_idx / config.workers_per_core;
    SetThreadAffinityMask(GetCurrentThread(), (static_cast<DWORD_PTR>(1) << core_id));
    at::set_num_threads(1);

    uint64_t local_step_cache = current_step.load(std::memory_order_relaxed);

    Logger logger("worker_" + std::to_string(worker_id), config.rl_dir, config.worker_logging_level);
    logger.rotate(local_step_cache, config.rotation_interval);
    
    logger.log("INFO", "=== GAME WORKER " + std::to_string(core_id) + " ===");

    std::atomic<int>* core_wait_count = core_wait_counts[core_index].get();
    
    // --- Coordinator takes ownership of the Engine ---
    chess::Board dummy;
    dummy.setFen(chess::constants::STARTPOS);
    MCTSEngine mcts(
        selector_config.node_pool_size, selector_config.batch_size_per_worker, 
        inference_queue, result_queues[logical_idx], logical_idx,
        selector_config.virtual_loss, selector_config.draw_cutoff, 
        selector_config.gumbel_c_visit, selector_config.gumbel_c_scale, 
        selector_config.gumbel_noise, dummy, std::vector<chess::Board>(), logger,
        shared_input_buffer, shared_policy_buffer, shared_value_buffer,
        buffer_free_slots, core_wait_count, config.workers_per_core
    );

    ActionSelector agent("worker_" + std::to_string(worker_id), logical_idx, selector_config, logger);

    while (!stop_event.load()) {
        local_step_cache = current_step.load(std::memory_order_relaxed);
        logger.rotate(local_step_cache, config.rotation_interval);

        int current_game_num = game_counter.fetch_add(1);
        
        char banner[512];
        snprintf(banner, sizeof(banner), "\n============================================================\n"
                                         "                    --- GAME %d STARTED ---\n"
                                         "============================================================", current_game_num);
        logger.log("CRITICAL", banner);

        chess::Board board;
        board.setFen(chess::constants::STARTPOS);
        agent.reset_for_new_game();

        bool game_over = false;
        int ply_count = 1;
        double final_game_value = 0.0;
        double game_entropy_sum = 0.0;
        std::string pgn_result = "*";
        
        std::vector<GameTransition> raw_training_data;
        raw_training_data.reserve(config.max_ply_length); // FIX: Pre-allocate capacity to stop heap shredding

        std::vector<chess::Board> history; 
        int total_input_size = model_config.input_planes * model_config.board_dim * model_config.board_dim;

        // FIX: Hoist mask allocation outside the simulation loop
        std::unique_ptr<bool[]> temp_mask(new bool[model_config.policy_moves]);

        while (!game_over && !stop_event.load()) {
            chess::Color current_turn = board.sideToMove();
            int move_number = ((ply_count - 1) / 2) + 1;
            std::string side_str = (current_turn == chess::Color::WHITE) ? "White" : "Black";
            
            char move_banner[512];
            snprintf(move_banner, sizeof(move_banner), 
                "\n============================================================\n"
                "                    --- MOVE %d: %s, PLY %d STARTED ---\n"
                "============================================================", 
                move_number, side_str.c_str(), ply_count);
            logger.log("INFO", move_banner);

            auto move_start_time = std::chrono::high_resolution_clock::now();

            // 1. Search
            mcts.reset(board, history);
            int sim_count = mcts.run_simulations(selector_config.gumbel_search_depth, selector_config.gumbel_m);
            double root_v_mix = mcts.root->calculate_v_mix();

            // 2. Generate Targets
            TargetResult targets = TargetGenerator::generate_targets(
                mcts.root, root_v_mix, board, selector_config, model_config, logger
            );

            // 3. Select Action
            SelectionResult move_result = agent.select_move(mcts.root, ply_count);

            auto move_end_time = std::chrono::high_resolution_clock::now();
            double total_move_time = std::chrono::duration<double>(move_end_time - move_start_time).count();
            double sim_speed = (total_move_time > 0) ? (sim_count / total_move_time) : 0.0;

            char timer_buffer[256];
            snprintf(timer_buffer, sizeof(timer_buffer), "Time: %.4fs | Speed: %.1f sim/s | Entropy: %.4f | Value: %.4f", 
                     total_move_time, sim_speed, targets.entropy, root_v_mix);
            logger.log("INFO", timer_buffer);

            if (move_result.resigned || move_result.best_move == chess::Move::NO_MOVE) {
                final_game_value = (current_turn == chess::Color::BLACK) ? 1.0 : -1.0; 
                pgn_result = (current_turn == chess::Color::WHITE) ? "0-1" : "1-0";
                game_over = true;
                logger.log("INFO", "Game ended by resignation.");
                break;
            }

            logger.log("INFO", "Selected move: " + chess::uci::moveToUci(move_result.best_move));

            game_entropy_sum += targets.entropy;

            GameTransition transition;
            transition.turn = current_turn;
            transition.move = move_result.best_move;
            transition.board_state.resize(total_input_size, 0);
            transition.policy = targets.policy_vector;
            
            board_to_tensor_69(board, history, transition.board_state.data());
            
            get_legal_move_mask(board, temp_mask.get()); // FIX: Use pre-allocated mask memory
            transition.legal_mask.clear();
            for(int i = 0; i < model_config.policy_moves; ++i) transition.legal_mask.push_back(temp_mask[i] ? 1 : 0);
            
            raw_training_data.push_back(transition);
            history.insert(history.begin(), board);
            if (history.size() > 4) history.pop_back();

            board.makeMove(move_result.best_move);
            ply_count++;

            auto result = board.isGameOver();
            if (result.second != chess::GameResult::NONE) {
                game_over = true;
                if (result.second == chess::GameResult::LOSE) {
                    if (board.sideToMove() == chess::Color::BLACK) {
                        final_game_value = 1.0; 
                        pgn_result = "1-0";
                    } else {
                        final_game_value = -1.0;
                        pgn_result = "0-1";
                    }
                } else {
                    final_game_value = 0.0;
                    pgn_result = "1/2-1/2";
                }
            } else if (ply_count >= config.max_ply_length) {
                game_over = true;
                final_game_value = 0.0;
                pgn_result = "1/2-1/2";
            }
        }
        
        _generate_pgn(current_game_num, raw_training_data, pgn_result, logger);

        CompletedGame game;
        game.game_number = current_game_num;
        game.transitions = std::move(raw_training_data); 
        game.final_game_value = final_game_value;
        game.local_step = local_step_cache;
        game.game_entropy_sum = game_entropy_sum;

        while (completed_games_queue.size() >= 200 && !stop_event.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        completed_games_queue.push(std::move(game));
    }
}