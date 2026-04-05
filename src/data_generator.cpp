#define NOMINMAX
#include "data_generator.hpp"
#include "logger.hpp"
#include <iostream>
#include <cstring>
#include <c10/util/Half.h>
#include "board_utils.hpp"
#include <fstream>
#include <string>
#include <windows.h> 

DataGenerator::DataGenerator(
    const YAML::Node& data_gen_cfg, const YAML::Node& mcts_cfg, const YAML::Node& sel_cfg, const YAML::Node& model_cfg,
    const std::string& rl_dir_in, const std::string& state_file_in, const std::string& db_path, int rot_interval,
    Logger& logger,
    std::vector<ThreadSafeQueue<std::vector<std::pair<int, int>>>*>& i_shards,
    std::vector<ThreadSafeQueue<std::vector<int>>>& r_queues,
    std::vector<torch::Tensor>& in_buffer, std::vector<torch::Tensor>& p_buffer, std::vector<torch::Tensor>& v_buffer,
    ThreadSafeQueue<int>& free_slots, std::atomic<size_t>& w_head, std::atomic<size_t>& b_count, size_t max_buf_size
) : main_logger(logger), lmdb_path(db_path),
    inference_shards(i_shards), result_queues(r_queues),
    shared_input_buffer(in_buffer), shared_policy_buffer(p_buffer), shared_value_buffer(v_buffer),
    buffer_free_slots(free_slots), write_head(w_head), buffer_count(b_count), max_buffer_size(max_buf_size),
    stop_event(false), game_counter(1) 
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
    config.state_file = state_file_in;
    config.rotation_interval = rot_interval;

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
    selector_config.minimax_smoothing_factor = mcts_cfg["minimax_smoothing_factor"].as<double>();
    selector_config.temperature_ply_cutoff = sel_cfg["temperature_ply_cutoff"].as<int>();
    selector_config.temperature_top_move = sel_cfg["temperature_top_move"].as<double>();
    selector_config.temperature_blunder_threshold = sel_cfg["temperature_blunder_threshold"].as<double>();
    selector_config.draw_cutoff = sel_cfg["draw_cutoff"].as<double>();
    selector_config.resignation_probability = sel_cfg["resignation_probability"].as<double>();
    selector_config.resignation_cutoff = sel_cfg["resignation_cutoff"].as<double>();

    mdb_env_create(&lmdb_env);
    mdb_env_set_mapsize(lmdb_env, (size_t)1024 * 1024 * 1024 * 128); 
    mdb_env_open(lmdb_env, lmdb_path.c_str(), MDB_NOSYNC | MDB_NOMETASYNC, 0664);

    main_logger.log("INFO", "DataGenerator successfully initialized.");
}

DataGenerator::~DataGenerator() {
    stop();
    if (lmdb_env) mdb_env_close(lmdb_env);
}

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

std::vector<uint8_t> DataGenerator::pack_bits(const std::vector<c10::Half>& data) {
    std::vector<uint8_t> out((data.size() + 7) / 8, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        // c10::Half supports comparison with 0.0f
        if (data[i] > 0.0f) out[i / 8] |= (1 << (7 - (i % 8)));
    }
    return out;
}

std::vector<uint8_t> DataGenerator::pack_bits_bool(const uint8_t* data, size_t size) {
    std::vector<uint8_t> out((size + 7) / 8, 0);
    for (size_t i = 0; i < size; ++i) {
        if (data[i]) out[i / 8] |= (1 << (7 - (i % 8)));
    }
    return out;
}

void DataGenerator::write_game_to_lmdb(const std::vector<GameTransition>& game_data, double final_game_value) {
    MDB_txn* txn;
    mdb_txn_begin(lmdb_env, nullptr, 0, &txn);
    MDB_dbi dbi;
    mdb_dbi_open(txn, nullptr, 0, &dbi);

    for (const auto& transition : game_data) {
        double value_target = (transition.turn == chess::Color::WHITE) ? final_game_value : -final_game_value;
        std::vector<uint8_t> p_board = pack_bits(transition.board_state);
        std::vector<uint8_t> p_mask = pack_bits_bool(transition.legal_mask.data(), model_config.policy_moves); 

        std::vector<uint16_t> indices;
        std::vector<uint16_t> values_fp16; 
        for (uint16_t i = 0; i < model_config.policy_moves; ++i) {
            if (transition.policy[i] > 0.0f) {
                indices.push_back(i);
                values_fp16.push_back(c10::Half(transition.policy[i]).x);
            }
        }

        uint16_t num_moves = indices.size();
        uint16_t target_fp16 = c10::Half(value_target).x;

        std::vector<uint8_t> blob;
        auto append_bytes = [&blob](const void* src, size_t size) {
            const uint8_t* bytes = static_cast<const uint8_t*>(src);
            blob.insert(blob.end(), bytes, bytes + size);
        };

        append_bytes(&num_moves, sizeof(num_moves));
        append_bytes(p_board.data(), p_board.size());
        append_bytes(p_mask.data(), p_mask.size());
        append_bytes(indices.data(), indices.size() * sizeof(uint16_t));
        append_bytes(values_fp16.data(), values_fp16.size() * sizeof(uint16_t));
        append_bytes(&target_fp16, sizeof(target_fp16));

        size_t head = write_head.fetch_add(1) % max_buffer_size;
        if (buffer_count.load() < max_buffer_size) buffer_count.fetch_add(1);

        std::string key_str = std::to_string(head);
        MDB_val key_val = { key_str.size(), (void*)key_str.data() };
        MDB_val data_val = { blob.size(), (void*)blob.data() };
        mdb_put(txn, dbi, &key_val, &data_val, 0);
    }
    mdb_txn_commit(txn);
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
        if (i % 12 == 0 && i != 0) {
            pgn << "\n";
        }
        
        if (i % 2 == 0) {
            pgn << (i / 2 + 1) << ". ";
        }

        pgn << chess::uci::moveToSan(temp_board, transitions[i].move) << " ";
        
        temp_board.makeMove(transitions[i].move);
    }
    
    pgn << result_str << "\n";

    logger.log("CRITICAL", "Game PGN:\n" + pgn.str());
}

void DataGenerator::worker_main(int logical_idx, int core_id) {
    int worker_id = logical_idx + 1;

    SetThreadAffinityMask(GetCurrentThread(), (static_cast<DWORD_PTR>(1) << core_id));
    at::set_num_threads(1);
    int local_step_cache = get_step_from_yaml(config.state_file, 0);

    Logger logger("game_worker_" + std::to_string(worker_id), config.rl_dir, config.worker_logging_level);
    logger.rotate(local_step_cache, config.rotation_interval);
    
    logger.log("INFO", "=== GAME WORKER " + std::to_string(core_id) + " ===");
    logger.log("INFO", "Worker " + std::to_string(worker_id) + " pinned to core: " + std::to_string(core_id));
    
    int num_shards = inference_shards.size();
    int shard_idx = logical_idx % num_shards; 
    
    ActionSelector agent("worker_" + std::to_string(worker_id), logical_idx, selector_config, 
                         model_config, logger, 
                         *inference_shards[shard_idx],
                         result_queues[logical_idx], 
                         shared_input_buffer, shared_policy_buffer, shared_value_buffer, buffer_free_slots);

    while (!stop_event.load()) {
        local_step_cache = get_step_from_yaml(config.state_file, local_step_cache);
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
        agent.set_name("step_" + std::to_string(local_step_cache));

        bool game_over = false;
        int ply_count = 1;
        double final_game_value = 0.0;
        std::string pgn_result = "*";
        std::vector<GameTransition> raw_training_data;
        std::vector<chess::Board> history; 
        int total_input_size = model_config.input_planes * model_config.board_dim * model_config.board_dim;

        while (!game_over && !stop_event.load()) {
            chess::Color current_turn = board.sideToMove();
            SelectionResult move_result = agent.select_action(board, history, ply_count, 
                                                              selector_config.gumbel_search_depth, 
                                                              selector_config.gumbel_m);

            if (move_result.resigned || move_result.best_move == chess::Move::NO_MOVE) {
                final_game_value = (current_turn == chess::Color::BLACK) ? 1.0 : -1.0; 
                pgn_result = (current_turn == chess::Color::WHITE) ? "0-1" : "1-0";
                game_over = true;
                logger.log("INFO", "Game ended by resignation.");
                break;
            }

            GameTransition transition;
            transition.turn = current_turn;
            transition.move = move_result.best_move;
            // FIXED: Using 0 instead of 0.0f
            transition.board_state.resize(total_input_size, 0);
            transition.policy = move_result.policy_vector;
            
            board_to_tensor_69(board, history, transition.board_state.data());
            
            std::unique_ptr<bool[]> temp_mask(new bool[model_config.policy_moves]);
            get_legal_move_mask(board, temp_mask.get());
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
                if (result.second == chess::GameResult::WIN) {
                    final_game_value = (board.sideToMove() == chess::Color::BLACK) ? 1.0 : -1.0;
                    pgn_result = (board.sideToMove() == chess::Color::BLACK) ? "1-0" : "0-1";
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
        // write_game_to_lmdb(raw_training_data, final_game_value);
    }
}