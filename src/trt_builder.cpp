#include "trt_builder.hpp"
#include <fstream>
#include <cuda_runtime_api.h>

std::unique_ptr<TRTBuilder::EngineResult> TRTBuilder::build_engine(const std::string& onnx_path, int max_batch_size, Logger& logger) {
    cudaSetDevice(0);

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gTRTLogger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gTRTLogger));

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) return nullptr;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 69, 8, 8});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{max_batch_size, 69, 8, 8});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{max_batch_size, 69, 8, 8});
    config->addOptimizationProfile(profile);

    if (builder->platformHasFastFp16()) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // kREFIT flag removed. TRT will now optimize more aggressively.
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512ULL * 1024 * 1024);

    logger.log("INFO", "TensorRT: Building engine from scratch...");
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) return nullptr;

    auto result = std::make_unique<EngineResult>();
    result->serialized_data.assign((uint8_t*)plan->data(), (uint8_t*)plan->data() + plan->size());
    return result;
}

void TRTBuilder::save_engine(const EngineResult& result, const std::string& engine_path) {
    std::ofstream outfile(engine_path, std::ios::binary);
    outfile.write((const char*)result.serialized_data.data(), result.serialized_data.size());
}