// =============================================================================
// main_trt_compile.cpp
//
// Standalone TensorRT engine compilation.
//
// Usage:
//   talbot_trt_compile <input.onnx> <output.engine> --input-planes N [options]
//
// Options:
//   --input-planes N    REQUIRED. Number of input planes for the profile.
//                       Must match the ONNX and your board_to_tensor.
//   --max-batch N       Max (and opt) batch size baked into the engine.
//                       MIN is always 1. Default: 256.
//   --force             Rebuild even if <output.engine> exists and is newer
//                       than <input.onnx>. Default: skip in that case.
//   -h, --help          Print this and exit 0.
//
// Exit codes:
//   0  success (built, or skipped because up-to-date)
//   1  bad arguments
//   2  input ONNX missing
//   3  build failed
//   4  writing output failed
//
// Notes:
//   * Engines are hardware- and TRT-version-specific. Cache produced here is
//     not portable to a different GPU generation or a different TRT major.
//   * A timing cache is maintained next to the ONNX at <onnx>.timing.cache;
//     that is TRTBuilder's existing behaviour and is preserved.
//   * Status messages go to stdout; parse/build errors go to stderr.
// =============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cstdlib>

#include "trt_builder.hpp"
#include "build_info.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string onnx_path;
    std::string engine_path;
    int  max_batch = 256;
    int  input_planes = -1;    // required; -1 = unset
    bool force = false;
};

void print_usage(std::ostream& os) {
    os <<
        "Usage: talbot_trt_compile <input.onnx> <output.engine> --input-planes N [options]\n"
        "\n"
        "Options:\n"
        "  --input-planes N   REQUIRED. Input plane count for the optimization profile.\n"
        "  --max-batch N      Max/opt batch size baked into the engine. Default: 256.\n"
        "  --force            Rebuild even if the engine is newer than the ONNX.\n"
        "  -h, --help         Print this help and exit.\n";
}

// Small hand-rolled parser -- one exe, three flags, no need for a library.
// Returns false on any parse error and prints the reason to stderr.
bool parse_args(int argc, char** argv, Args& out) {
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(std::cout);
            std::exit(0);
        } else if (a == "--force") {
            out.force = true;
        } else if (a == "--max-batch") {
            if (i + 1 >= argc) { std::cerr << "error: --max-batch requires a value\n"; return false; }
            try { out.max_batch = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "error: --max-batch value is not an integer\n"; return false; }
            if (out.max_batch < 1) { std::cerr << "error: --max-batch must be >= 1\n"; return false; }
        } else if (a == "--input-planes") {
            if (i + 1 >= argc) { std::cerr << "error: --input-planes requires a value\n"; return false; }
            try { out.input_planes = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "error: --input-planes value is not an integer\n"; return false; }
            if (out.input_planes < 1) { std::cerr << "error: --input-planes must be >= 1\n"; return false; }
        } else if (a.rfind("--", 0) == 0 || (!a.empty() && a[0] == '-' && a != "-")) {
            std::cerr << "error: unknown flag '" << a << "'\n";
            return false;
        } else {
            positional.push_back(a);
        }
    }
    if (positional.size() != 2) {
        std::cerr << "error: expected exactly 2 positional args (onnx, engine); got "
                  << positional.size() << "\n";
        return false;
    }
    if (out.input_planes < 0) {
        std::cerr << "error: --input-planes is required\n";
        return false;
    }
    out.onnx_path   = positional[0];
    out.engine_path = positional[1];
    return true;
}

// `make`-style freshness check: skip if the engine exists and its mtime is >=
// the ONNX's. Any filesystem error while checking -> conservative rebuild.
bool engine_is_up_to_date(const std::string& onnx, const std::string& engine) {
    std::error_code ec;
    if (!fs::exists(engine, ec) || ec) return false;
    auto engine_mtime = fs::last_write_time(engine, ec);
    if (ec) return false;
    auto onnx_mtime = fs::last_write_time(onnx, ec);
    if (ec) return false;
    return engine_mtime >= onnx_mtime;
}

}  // namespace

int main(int argc, char** argv) {
    std::cerr << "Talbot [git " << talbot::GIT_COMMIT
            << "] built " << talbot::BUILD_TIME << "\n";

    Args args;
    if (!parse_args(argc, argv, args)) {
        std::cerr << "\n";
        print_usage(std::cerr);
        return 1;
    }

    // Missing input is an error, not a skip.
    std::error_code ec;
    if (!fs::exists(args.onnx_path, ec) || ec) {
        std::cerr << "error: input ONNX not found: " << args.onnx_path << "\n";
        return 2;
    }

    // Freshness cache. Cheap early-out so callers (training loop, eval harness)
    // can shell out unconditionally and pay ~0ms when nothing changed.
    if (!args.force && engine_is_up_to_date(args.onnx_path, args.engine_path)) {
        std::cout << "up-to-date, skipping: " << args.engine_path
                  << " (use --force to rebuild)\n";
        return 0;
    }

    // Make sure the output directory exists so a fresh build doesn't fail for
    // a stupid reason. Non-fatal if it already exists.
    fs::path out_dir = fs::path(args.engine_path).parent_path();
    if (!out_dir.empty()) {
        fs::create_directories(out_dir, ec);
    }

    auto start = std::chrono::steady_clock::now();

    auto result = TRTBuilder::build_engine(
        args.onnx_path,
        args.max_batch,
        args.input_planes
    );

    if (!result) {
        std::cerr << "error: TensorRT build failed\n";
        return 3;
    }

    // save_engine doesn't report failure via return value, so verify the file
    // materialised at a non-zero size before claiming success.
    TRTBuilder::save_engine(*result, args.engine_path);
    auto written = fs::exists(args.engine_path, ec) ? fs::file_size(args.engine_path, ec) : 0;
    if (ec || written == 0) {
        std::cerr << "error: failed to write engine to " << args.engine_path << "\n";
        return 4;
    }

    auto dur = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "built " << args.engine_path
              << " (" << written << " bytes) in "
              << dur << "s\n";
    return 0;
}