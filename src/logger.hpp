#pragma once
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <mutex>
#include <algorithm>

inline std::mutex console_mutex;

inline int get_step_from_yaml(const std::string& filepath, int last_known_step) {
    // ... exactly the same ...
    std::ifstream file(filepath);
    if (!file.is_open()) return last_known_step;
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("training_steps:");
        if (pos != std::string::npos) {
            try { return std::stoi(line.substr(pos + 15)); } catch (...) { return last_known_step; }
        }
    }
    return last_known_step;
}

class Logger {
private:
    std::string name;
    std::ofstream log_file;
    int current_rotation_step = -1;
    std::string rl_dir;
    int min_log_level; // Added

    int parse_level(std::string level_str) {
        std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::toupper);
        if (level_str == "DEBUG") return 10;
        if (level_str == "INFO") return 20;
        if (level_str == "WARNING") return 30;
        if (level_str == "ERROR") return 40;
        if (level_str == "CRITICAL") return 50;
        return 20; // Default INFO
    }

public:
    // Added min_log_level to constructor
    Logger(const std::string& name, const std::string& rl_dir, int min_log_level = 20) 
        : name(name), rl_dir(rl_dir), min_log_level(min_log_level) {}
    
    ~Logger() {
        if (log_file.is_open()) log_file.close();
    }

    void rotate(int global_step, int rotation_interval) {
        // ... exactly the same ...
        int target_folder_step = (global_step / rotation_interval) * rotation_interval;
        if (target_folder_step != current_rotation_step) {
            if (log_file.is_open()) log_file.close();
            char folder_name[256];
            snprintf(folder_name, sizeof(folder_name), "%s/run_step_%06d", rl_dir.c_str(), target_folder_step);
            std::filesystem::create_directories(folder_name);
            std::string log_path = std::string(folder_name) + "/" + name + ".log";
            log_file.open(log_path, std::ios::app);
            current_rotation_step = target_folder_step;
        }
    }

    void log(const std::string& level, const std::string& message) {
        // Filter out logs below the configured threshold
        if (parse_level(level) < min_log_level) return;

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") 
           << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
           << "[" << name << "] [" << level << "] " << message;
           
        {
            std::lock_guard<std::mutex> lock(console_mutex);
        }

        if (log_file.is_open()) {
            log_file << ss.str() << "\n";
            log_file.flush();
        }
    }
};