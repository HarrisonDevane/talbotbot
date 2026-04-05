#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include "logger.hpp"

// Custom Logger to capture TensorRT internal optimization messages
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only log warnings and errors to keep the console clean
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT-BUILD] " << msg << std::endl;
        }
    }
} inline gTRTLogger;

class TRTBuilder {
public:
    struct EngineResult {
        std::vector<uint8_t> serialized_data;
    };

    // Main entry point for the background thread
    static std::unique_ptr<EngineResult> build_engine(
        const std::string& onnx_path, 
        int max_batch_size, 
        Logger& logger
    );

    // Helper to save the binary to disk (so the next boot is instant)
    static void save_engine(const EngineResult& result, const std::string& engine_path);
};