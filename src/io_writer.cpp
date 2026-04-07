#define NOMINMAX
#include "io_writer.hpp"
#include <iostream>
#include <windows.h> 
#include <cmath>
#include <cstring>
#include <deque>

IOWriter::IOWriter(
    MDB_env* env, size_t min_buffer, size_t max_buffer, int ramp_steps,
    const std::vector<int>& core_ids, const std::string& rl_dir, 
    size_t flush_threshold, int logging_level, int rot_interval, const ModelConfig& model_cfg, 
    ThreadSafeQueue<CompletedGame>& queue, std::atomic<uint64_t>& step_ref, std::atomic<size_t>& w_head, 
    std::atomic<size_t>& b_count, std::atomic<size_t>& b_wraps,
    size_t start_games, size_t start_samples, double start_entropy
) : lmdb_env(env), min_buffer_size(min_buffer), max_buffer_size(max_buffer),
    buffer_ramp_steps(ramp_steps), io_cores(core_ids), 
    flush_threshold(flush_threshold), rotation_interval(rot_interval), 
    logger("io_writer", rl_dir, logging_level), model_config(model_cfg), 
    completed_games_queue(queue), current_step(step_ref), write_head(w_head), 
    buffer_count(b_count), buffer_wraps(b_wraps),
    lifetime_games(start_games), lifetime_samples(start_samples), lifetime_entropy(start_entropy)
{
    logger.rotate(current_step.load(std::memory_order_relaxed), rotation_interval);
}

IOWriter::~IOWriter() {
    stop();
}

void IOWriter::start() {
    writer_thread = std::thread(&IOWriter::run, this);
}

void IOWriter::stop() {
    stop_event.store(true);
    if (writer_thread.joinable()) writer_thread.join();
}

size_t IOWriter::get_dynamic_buffer_limit(int current_step) {
    if (current_step >= buffer_ramp_steps) return max_buffer_size;
    double progress = static_cast<double>(current_step) / static_cast<double>(buffer_ramp_steps);
    double growth_range = static_cast<double>(max_buffer_size - min_buffer_size);
    return min_buffer_size + static_cast<size_t>(progress * growth_range);
}

std::vector<uint8_t> IOWriter::pack_bits(const std::vector<c10::Half>& data) {
    std::vector<uint8_t> out((data.size() + 7) / 8, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] > 0.0f) out[i / 8] |= (1 << (7 - (i % 8)));
    }
    return out;
}

std::vector<uint8_t> IOWriter::pack_bits_bool(const uint8_t* data, size_t size) {
    std::vector<uint8_t> out((size + 7) / 8, 0);
    for (size_t i = 0; i < size; ++i) {
        if (data[i]) out[i / 8] |= (1 << (7 - (i % 8)));
    }
    return out;
}

void IOWriter::run() {
    if (!io_cores.empty()) {
        DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << io_cores[0]);
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
    at::set_num_threads(1);

    std::deque<CompletedGame> game_buffer;
    size_t current_buffered_samples = 0;
    size_t current_game_transition_idx = 0;
    
    size_t uncommitted_games = 0;
    double uncommitted_entropy = 0.0;

    while (!stop_event.load()) {
        CompletedGame game;
        if (completed_games_queue.try_pop(game)) {
            
            current_buffered_samples += game.transitions.size();
            game_buffer.push_back(std::move(game));

            if (current_buffered_samples >= flush_threshold) {
                uint64_t local_step = current_step.load(std::memory_order_relaxed);
                logger.rotate(local_step, rotation_interval);

                MDB_txn* txn;
                mdb_txn_begin(lmdb_env, nullptr, 0, &txn);
                MDB_dbi dbi;
                mdb_dbi_open(txn, nullptr, 0, &dbi);

                size_t dynamic_max = get_dynamic_buffer_limit(local_step);
                size_t current_head = write_head.load(std::memory_order_relaxed);
                size_t current_cnt = buffer_count.load(std::memory_order_relaxed);
                size_t current_wraps = buffer_wraps.load(std::memory_order_relaxed);

                size_t samples_to_write = flush_threshold;

                while (samples_to_write > 0 && !game_buffer.empty()) {
                    auto& g = game_buffer.front();
                    size_t available_in_game = g.transitions.size() - current_game_transition_idx;
                    size_t chunk_size = std::min(samples_to_write, available_in_game);

                    for (size_t i = 0; i < chunk_size; ++i) {
                        const auto& transition = g.transitions[current_game_transition_idx + i];
                        double value_target = (transition.turn == chess::Color::WHITE) ? g.final_game_value : -g.final_game_value;
                        std::vector<uint8_t> p_board = pack_bits(transition.board_state);
                        std::vector<uint8_t> p_mask = pack_bits_bool(transition.legal_mask.data(), model_config.policy_moves); 

                        std::vector<uint16_t> indices;
                        std::vector<uint16_t> values_fp16; 
                        for (uint16_t j = 0; j < model_config.policy_moves; ++j) {
                            if (transition.policy[j] > 0.0f) {
                                indices.push_back(j);
                                values_fp16.push_back(c10::Half(transition.policy[j]).x);
                            }
                        }

                        uint16_t num_moves = indices.size();
                        uint16_t target_fp16 = c10::Half(value_target).x;

                        std::vector<uint8_t> raw_payload;
                        auto append_bytes = [&raw_payload](const void* src, size_t size) {
                            const uint8_t* bytes = static_cast<const uint8_t*>(src);
                            raw_payload.insert(raw_payload.end(), bytes, bytes + size);
                        };

                        append_bytes(&num_moves, sizeof(num_moves));
                        append_bytes(p_board.data(), p_board.size());
                        append_bytes(p_mask.data(), p_mask.size());
                        append_bytes(indices.data(), indices.size() * sizeof(uint16_t));
                        append_bytes(values_fp16.data(), values_fp16.size() * sizeof(uint16_t));
                        append_bytes(&target_fp16, sizeof(target_fp16));

                        size_t max_dst_size = ZSTD_compressBound(raw_payload.size());
                        std::vector<uint8_t> compressed_buf(max_dst_size);
                        size_t compressed_size = ZSTD_compress(compressed_buf.data(), max_dst_size, raw_payload.data(), raw_payload.size(), 1);
                        if (ZSTD_isError(compressed_size)) continue; 

                        std::string key_str = std::to_string(current_head);
                        MDB_val key_val = { key_str.size(), (void*)key_str.data() };
                        MDB_val data_val = { compressed_size, (void*)compressed_buf.data() };
                        mdb_put(txn, dbi, &key_val, &data_val, 0);

                        if (current_cnt < dynamic_max) current_cnt++;
                        current_head = (current_head + 1) % dynamic_max;
                        if (current_head == 0) current_wraps++;
                    }

                    current_game_transition_idx += chunk_size;
                    samples_to_write -= chunk_size;

                    // If the entire game is consumed, advance queue
                    if (current_game_transition_idx == g.transitions.size()) {
                        uncommitted_games++;
                        uncommitted_entropy += g.game_entropy_sum;
                        game_buffer.pop_front();
                        current_game_transition_idx = 0;
                    }
                }

                current_buffered_samples -= flush_threshold;
                lifetime_games += uncommitted_games;
                lifetime_samples += flush_threshold;

                double raw_entropy = lifetime_entropy + uncommitted_entropy;
                lifetime_entropy = std::round(raw_entropy * 100.0) / 100.0;

                // Reset trackers for next batch of full games
                uncommitted_games = 0;
                uncommitted_entropy = 0.0;

                CppState state;
                state.games_played = lifetime_games;
                state.samples_generated = lifetime_samples;
                state.lifetime_entropy = lifetime_entropy;
                state.buffer_count = current_cnt;
                state.buffer_head_ptr = current_head;
                state.buffer_wraps = current_wraps;

                MDB_val state_key = { 11, (void*)"__CPP_STATE" };
                MDB_val state_val = { sizeof(CppState), &state };
                mdb_put(txn, dbi, &state_key, &state_val, 0);

                mdb_txn_commit(txn);

                write_head.store(current_head, std::memory_order_relaxed);
                buffer_count.store(current_cnt, std::memory_order_relaxed);
                buffer_wraps.store(current_wraps, std::memory_order_relaxed);

                logger.log("INFO", "Successfully flushed exactly " + std::to_string(flush_threshold) + " samples. LMDB Count: " + std::to_string(current_cnt));
            }
        }
    }
}