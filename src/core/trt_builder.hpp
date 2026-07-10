#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include "logger.hpp"

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
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

    static std::unique_ptr<EngineResult> build_engine(
        const std::string& onnx_path, 
        int max_batch_size,
        int input_planes,
        Logger& logger
    );

    static void save_engine(const EngineResult& result, const std::string& engine_path);
};