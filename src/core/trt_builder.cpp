#include "trt_builder.hpp"
#include <fstream>
#include <vector>
#include <cuda_runtime_api.h>

namespace {
// Read an entire file into a byte buffer. Returns empty on any failure
// (missing file is the normal "cold cache" case, not an error).
std::vector<char> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    if (size <= 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!f.read(buf.data(), size)) return {};
    return buf;
}
}  // namespace

std::unique_ptr<TRTBuilder::EngineResult> TRTBuilder::build_engine(const std::string& onnx_path, int max_batch_size, int input_planes, Logger& logger) {
    cudaSetDevice(0);

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gTRTLogger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gTRTLogger));

    logger.log("INFO", "TensorRT: Parsing ONNX file...");
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        logger.log("ERROR", "TensorRT: Failed to parse ONNX file");
        return nullptr;
    }
    logger.log("INFO", "TensorRT: ONNX parsing complete.");

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, input_planes, 8, 8});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{max_batch_size, input_planes, 8, 8});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{max_batch_size, input_planes, 8, 8});
    config->addOptimizationProfile(profile);

    if (builder->platformHasFastFp16()) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512ULL * 1024 * 1024);

    // ---- Timing cache (load) ------------------------------------------------
    // Reuse tactic-selection results from previous builds so the builder skips
    // the live hardware profiling phase. The cache is a separate file next to
    // the ONNX; missing/empty -> cold build that re-profiles, then writes it.
    // ignoreMismatch=false makes TRT reject a cache from a different GPU / TRT
    // version, so a stale file is safely regenerated rather than misused.
    const std::string cache_path = onnx_path + ".timing.cache";
    std::vector<char> cache_blob = read_file_bytes(cache_path);
    std::unique_ptr<nvinfer1::ITimingCache> timing_cache(
        config->createTimingCache(cache_blob.data(), cache_blob.size()));
    if (timing_cache) {
        if (!config->setTimingCache(*timing_cache, /*ignoreMismatch=*/false)) {
            logger.log("WARN", "TensorRT: setTimingCache failed; building without cache.");
        } else if (cache_blob.empty()) {
            logger.log("INFO", "TensorRT: timing cache cold (no usable file); will profile and write.");
        } else {
            logger.log("INFO", "TensorRT: timing cache loaded (" +
                       std::to_string(cache_blob.size()) + " bytes).");
        }
    } else {
        logger.log("WARN", "TensorRT: createTimingCache returned null; building without cache.");
    }
    // -------------------------------------------------------------------------

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    
    if (!plan) {
        logger.log("ERROR", "TensorRT: Engine build failed");
        return nullptr;
    }

    // ---- Timing cache (serialize) -------------------------------------------
    // Persist the (possibly newly-populated) cache so the next build -- and
    // crucially the next *process run* -- reuses it instead of profiling cold.
    if (timing_cache) {
        std::unique_ptr<nvinfer1::IHostMemory> serialized(timing_cache->serialize());
        if (serialized && serialized->size() > 0) {
            std::ofstream out(cache_path, std::ios::binary);
            if (out) {
                out.write(reinterpret_cast<const char*>(serialized->data()),
                          static_cast<std::streamsize>(serialized->size()));
                logger.log("INFO", "TensorRT: timing cache written (" +
                           std::to_string(serialized->size()) + " bytes).");
            } else {
                logger.log("WARN", "TensorRT: could not open timing cache for writing: " + cache_path);
            }
        }
    }
    // -------------------------------------------------------------------------

    auto result = std::make_unique<EngineResult>();
    result->serialized_data.assign(
        reinterpret_cast<uint8_t*>(plan->data()),
        reinterpret_cast<uint8_t*>(plan->data()) + plan->size()
    );
    
    logger.log("INFO", "TensorRT: Engine build complete.");
    return result;
}

void TRTBuilder::save_engine(const EngineResult& result, const std::string& engine_path) {
    std::ofstream outfile(engine_path, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(result.serialized_data.data()), result.serialized_data.size());
}