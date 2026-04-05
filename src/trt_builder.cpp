#include "trt_builder.hpp"
#include <fstream>

std::unique_ptr<TRTBuilder::EngineResult> TRTBuilder::build_engine(
    const std::string& onnx_path, 
    int max_batch_size, 
    Logger& logger
) {
    // 1. Initialize Builder, Network, and Parser
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gTRTLogger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gTRTLogger));

    // 2. Parse the ONNX file exported from Python
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        logger.log("CRITICAL", "TRT Builder: Failed to parse ONNX file.");
        return nullptr;
    }

    // 3. Configure the Build
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // Set up Optimization Profile for Dynamic Batching
    auto profile = builder->createOptimizationProfile();
    // Input name must match the name in your Python onnx.export
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 69, 8, 8});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{max_batch_size, 69, 8, 8});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{max_batch_size, 69, 8, 8});
    config->addOptimizationProfile(profile);

    // 4. Enable FP16 (Tensor Cores) for maximum speed
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 5. Set Memory Pool (Workspace) for kernel benchmarking
    // Allocating 2GB allows TRT to find the fastest possible kernels
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL * 1024 * 1024 * 1024);

    // 6. Build and Serialize (The "Cooking" phase)
    logger.log("INFO", "TensorRT: Starting engine build... (This takes ~45-60s)");
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    
    if (!plan) {
        logger.log("CRITICAL", "TensorRT build failed to generate a plan.");
        return nullptr;
    }

    auto result = std::make_unique<EngineResult>();
    result->serialized_data.assign((uint8_t*)plan->data(), (uint8_t*)plan->data() + plan->size());
    return result;
}

void TRTBuilder::save_engine(const EngineResult& result, const std::string& engine_path) {
    std::ofstream outfile(engine_path, std::ios::binary);
    outfile.write((const char*)result.serialized_data.data(), result.serialized_data.size());
}