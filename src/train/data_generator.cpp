#define NOMINMAX
#include "data_generator.hpp"
#include "logger.hpp"
#include <iostream>
#include <cstring>
#include <c10/util/Half.h>
#include "board_utils.hpp"
#include "pgn_writer.hpp"
#include <sstream>
#include <windows.h> 
#include <cmath>
#include <random>

DataGenerator::DataGenerator(
    const YAML::Node& pool_cfg,
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
    config.adjudication_max_ply_length = data_gen_cfg["adjudication_max_ply_length"].as<int>();
    config.adjudication_draw_threshold = data_gen_cfg["adjudication_draw_threshold"].as<double>();
    config.adjudication_draw_plies     = data_gen_cfg["adjudication_draw_plies"].as<int>();
    config.adjudication_draw_min_move  = data_gen_cfg["adjudication_draw_min_move"].as<int>();
    config.adjudication_draw_probability    = data_gen_cfg["adjudication_draw_probability"].as<double>();
    config.worker_logging_level = data_gen_cfg["worker_logging_level"].as<int>();
    config.target_shrinkage_k = data_gen_cfg["target_shrinkage_k"].as<double>();
    config.rl_dir = rl_dir_in;
    config.rotation_interval = rot_interval;

    for (int i = 0; i < config.num_cores; ++i) {
        core_wait_counts.push_back(std::make_unique<std::atomic<int>>(0));
    }

    model_config.input_planes = model_cfg["model"]["input_planes"].as<int>();
    model_config.board_dim = model_cfg["model"]["board_dim"].as<int>();
    model_config.policy_moves = model_cfg["model"]["total_policy_moves"].as<int>();

    // Single loader populates both configs. Shared contempt/draw_cutoff
    // fields cannot drift; adding a new mcts or selection knob is a one-line
    // change in load_configs() that both training and UCI pick up.
    LoadedConfigs loaded = load_configs(mcts_cfg, sel_cfg, /*require_gumbel_m=*/true);
    mcts_config     = loaded.mcts;
    selector_config = loaded.selector;

    // --- Pool sizing block (required) ---------------------------------------
    // Fed to MCTSEngine::pool_sizing_cfg post-construction. Same 5 knobs
    // as play_uci.yaml but typically with smaller caps -- training only ever
    // runs gumbel_search_depth-sim searches.
    pool_sizing_cfg.avg_branching       = pool_cfg["avg_branching"].as<double>();
    pool_sizing_cfg.node_safety_factor  = pool_cfg["node_safety_factor"].as<double>();
    pool_sizing_cfg.edge_safety_factor  = pool_cfg["edge_safety_factor"].as<double>();
    pool_sizing_cfg.node_hard_cap_bytes = (size_t)pool_cfg["node_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;
    pool_sizing_cfg.edge_hard_cap_bytes = (size_t)pool_cfg["edge_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;

    main_logger.log("INFO", "DataGenerator logic loop initialized.");
}

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

    // Training runs an exactly-known number of sims per search
    // (gumbel_search_depth). Size the initial pools for that -- the first
    // reset() then finds capacity already sufficient and doesn't grow.
    PoolTargets initial_targets = MCTSEngine::predict_pool_needs_static(
        mcts_config.gumbel_search_depth, pool_sizing_cfg);

    MCTSEngine mcts(
        mcts_config,
        static_cast<int>(initial_targets.node_target),
        static_cast<int>(initial_targets.edge_target),
        inference_queue, result_queues[logical_idx], logical_idx,
        dummy, std::vector<chess::Board>(), logger,
        shared_input_buffer, shared_policy_buffer, shared_value_buffer,
        buffer_free_slots, core_wait_count, config.workers_per_core, false
    );
    mcts.pool_sizing_cfg = pool_sizing_cfg;

    ActionSelector agent("worker_" + std::to_string(worker_id), logical_idx, selector_config, logger);

    // Per-worker RNG for adjudication-enable roll (once per game, matching the
    // resignation-probability pattern). Seeded from worker_id so runs are
    // reproducible per worker while workers get different sequences.
    std::mt19937 adj_rng(static_cast<uint32_t>(worker_id) * 2654435761u);
    std::uniform_real_distribution<double> adj_dist(0.0, 1.0);

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
        // Draw adjudication windowed counter: consecutive plies where both
        // (a) move number >= min, (b) move was quiet (no capture, no pawn),
        // (c) root D probability >= threshold. Resets on any failure.
        int consecutive_draw_plies = 0;
        // Per-game adjudication-enable roll. Matches the resignation-probability
        // pattern -- some fraction of games act as calibration samples that
        // play out fully even when adjudication criteria are met, preventing
        // the feedback-loop bias where an increasingly-confident net adjudicates
        // more aggressively and never sees positions play out.
        const bool adjudication_enabled_this_game =
            adj_dist(adj_rng) < config.adjudication_draw_probability;
        
        std::vector<GameTransition> raw_training_data;
        raw_training_data.reserve(config.adjudication_max_ply_length); // FIX: Pre-allocate capacity to stop heap shredding

        std::vector<chess::Board> history; 
        int total_input_size = model_config.input_planes * model_config.board_dim * model_config.board_dim;

        // FIX: Hoist mask allocation outside the simulation loop
        std::unique_ptr<bool[]> temp_mask(new bool[model_config.policy_moves]);

        while (!game_over && !stop_event.load()) {
            chess::Color current_turn = board.sideToMove();
            int move_number = ((ply_count - 1) / 2) + 1;
            std::string side_str = (current_turn == chess::Color::WHITE) ? "White" : "Black";

            auto move_start_time = std::chrono::high_resolution_clock::now();

            // 1. Search
            //   Pool targets from the exact-known sim budget. After move 1
            //   this is a no-op grow (capacity already sufficient).
            PoolTargets pt = mcts.predict_pool_needs(mcts_config.gumbel_search_depth);
            mcts.reset(board, history, pt.node_target, pt.edge_target);
            int sim_count = mcts.run_simulations_fixed(mcts_config.gumbel_search_depth, mcts_config.gumbel_m);
            double root_v_mix = mcts.root->calculate_v_mix(mcts_config.contempt);

            // 2. Generate Targets
            TargetResult targets = TargetGenerator::generate_targets(
                mcts.root, board, mcts_config, model_config, config.target_shrinkage_k, logger
            );

            // 3. Select Action
            SelectionResult move_result = agent.select_move(mcts.root, ply_count, &mcts);

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

            // --- Draw adjudication (windowed root-D check) ---
            // The selected move hasn't been played yet, so isCapture / piece-type
            // reads are against the current pre-move board. Root D-probability
            // comes from the search that just completed for this position.
            {
                double root_d = 0.0;
                if (mcts.root->visits > 0) {
                    root_d = static_cast<double>(mcts.root->d_sum) / mcts.root->visits;
                }
                const bool is_quiet =
                    !board.isCapture(move_result.best_move)
                    && board.at(move_result.best_move.from()).type() != chess::PieceType::PAWN;

                if (move_number >= config.adjudication_draw_min_move
                    && is_quiet
                    && root_d >= config.adjudication_draw_threshold) {
                    consecutive_draw_plies++;
                } else {
                    consecutive_draw_plies = 0;
                }

                if (consecutive_draw_plies >= config.adjudication_draw_plies
                    && adjudication_enabled_this_game) {
                    game_over = true;
                    final_game_value = 0.0;
                    pgn_result = "1/2-1/2";
                    char buf[192];
                    snprintf(buf, sizeof(buf),
                        "Game adjudicated as draw: root_d=%.3f threshold=%.3f "
                        "plies=%d move=%d",
                        root_d, config.adjudication_draw_threshold,
                        consecutive_draw_plies, move_number);
                    logger.log("INFO", buf);
                    break;
                }
            }

            logger.log("INFO", "Selected move: " + chess::uci::moveToUci(move_result.best_move));

            game_entropy_sum += targets.entropy;

            GameTransition transition;
            transition.turn = current_turn;
            transition.move = move_result.best_move;
            transition.board_state.resize(total_input_size, 0);
            transition.policy = targets.policy_vector;
            
            board_to_tensor(board, history, transition.board_state.data());
            
            get_legal_move_mask(board, temp_mask.get()); // FIX: Use pre-allocated mask memory
            transition.legal_mask.clear();
            for(int i = 0; i < model_config.policy_moves; ++i) transition.legal_mask.push_back(temp_mask[i] ? 1 : 0);
            
            raw_training_data.push_back(transition);
            history.insert(history.begin(), board);
            if (history.size() > 7) history.pop_back();

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
            } else if (ply_count >= config.adjudication_max_ply_length) {
                game_over = true;
                final_game_value = 0.0;
                pgn_result = "1/2-1/2";
            }
        }
        
        // Build minimal PGN via shared writer. Training runs millions of games,
        // so we skip per-move annotations, date, time-control tags -- just the
        // Seven Tag Roster + ECO + moves + result.
        PgnHeader hdr;
        hdr.event = "Self-Play Game " + std::to_string(current_game_num);
        hdr.site  = "Talbot C++ Engine";
        hdr.white = "Talbot Agent";
        hdr.black = "Talbot Agent";

        std::vector<chess::Move> game_moves;
        game_moves.reserve(raw_training_data.size());
        for (const auto& t : raw_training_data) game_moves.push_back(t.move);

        const std::string pgn = build_pgn(hdr, game_moves, {}, pgn_result, PgnConfig::minimal());
        logger.log("CRITICAL", "Game PGN:\n" + pgn);

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