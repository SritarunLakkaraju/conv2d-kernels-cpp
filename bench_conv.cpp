// bench_conv.cpp - compares the convolution algorithms and measures what
// fusing conv + bias + ReLU into a single pass buys.
//
// usage: bench_conv [results_dir]
// Everything here is single-threaded so that the comparison is between
// algorithms, not between thread counts.
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "conv.h"

struct Layer {
    const char* name;
    conv::Shape s;
};

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    const std::vector<Layer> layers = {
        {"3->64 @224", {3, 224, 224, 64, 3, 3, 1, 1}},
        {"64->64 @56", {64, 56, 56, 64, 3, 3, 1, 1}},
        {"128->128 @28", {128, 28, 28, 128, 3, 3, 1, 1}},
        {"256->256 @14", {256, 14, 14, 256, 3, 3, 1, 1}},
        {"512->512 @7", {512, 7, 7, 512, 3, 3, 1, 1}},
    };
    // Extra shapes for the fusion study only: few input channels and a large
    // output, i.e. little arithmetic per output element.
    const std::vector<Layer> fusion_only = {
        {"3->32 @512", {3, 512, 512, 32, 3, 3, 1, 1}},
        {"1->16 @1024", {1, 1024, 1024, 16, 3, 3, 1, 1}},
    };

    FILE* csv = dir.empty() ? nullptr : std::fopen((dir + "/conv_algos.csv").c_str(), "w");
    if (csv) std::fprintf(csv, "layer,algo,ms,gflops,max_abs_err\n");
    FILE* fcsv = dir.empty() ? nullptr : std::fopen((dir + "/conv_fusion.csv").c_str(), "w");
    if (fcsv) std::fprintf(fcsv, "layer,out_mb,separate_ms,fused_ms,speedup\n");

    std::printf("3x3 convolution, stride 1, pad 1, single thread. ms (effective GFLOPS)\n");
    std::printf("%-14s %18s %18s %18s %18s\n", "layer", "direct naive", "direct NEON", "im2col+GEMM", "Winograd");

    struct FusionRow {
        std::string name;
        double out_mb, sep, fused;
    };
    std::vector<FusionRow> fusion;

    util::Rng rng(1);
    for (const Layer& L : layers) {
        const conv::Shape& s = L.s;
        const size_t in_n = (size_t)s.IC * s.H * s.W, w_n = (size_t)s.OC * s.IC * 9;
        const int hw = s.OH() * s.OW();
        const size_t out_n = (size_t)s.OC * hw;
        util::avec<float> in(in_n), w(w_n), bias(s.OC), ref(out_n), out(out_n), scratch;
        util::fill_uniform(in.data(), in_n, rng);
        util::fill_uniform(w.data(), w_n, rng, -0.1f, 0.1f);
        util::fill_uniform(bias.data(), s.OC, rng);
        std::vector<float> U;
        conv::winograd_transform_kernel(s, w.data(), U);
        conv::WinogradScratch ws;

        conv::direct(s, in.data(), w.data(), nullptr, ref.data());

        struct Algo {
            const char* name;
            std::function<void()> run;
        };
        const std::vector<Algo> algos = {
            {"direct_naive", [&] { conv::direct(s, in.data(), w.data(), nullptr, out.data()); }},
            {"direct_neon", [&] { conv::direct3x3(s, in.data(), w.data(), nullptr, false, out.data(), scratch); }},
            {"im2col_gemm", [&] { conv::im2col_gemm(s, in.data(), w.data(), out.data(), scratch); }},
            {"winograd", [&] { conv::winograd3x3(s, in.data(), U.data(), out.data(), ws); }},
        };
        std::printf("%-14s", L.name);
        for (const Algo& a : algos) {
            std::fill(out.begin(), out.end(), -1.0f);
            const double t = util::time_best(a.run);
            const double err = util::max_abs_diff(out.data(), ref.data(), out_n);
            const double gf = s.flops() / t * 1e-9;
            std::printf(" %9.3f (%6.1f)", t * 1e3, gf);
            if (err > 2e-3) std::printf("[ERR %.1e]", err);
            if (csv) std::fprintf(csv, "%s,%s,%.5f,%.2f,%.3e\n", L.name, a.name, t * 1e3, gf, err);
        }
        std::printf("\n");

    }

    // Fusion: identical convolution kernel, with bias and ReLU either as two
    // further sweeps over the output tensor or folded into the conv's own store.
    std::vector<Layer> fusion_layers = layers;
    fusion_layers.insert(fusion_layers.begin(), fusion_only.begin(), fusion_only.end());
    for (const Layer& L : fusion_layers) {
        const conv::Shape& s = L.s;
        const size_t in_n = (size_t)s.IC * s.H * s.W, w_n = (size_t)s.OC * s.IC * 9;
        const int hw = s.OH() * s.OW();
        const size_t out_n = (size_t)s.OC * hw;
        util::avec<float> in(in_n), w(w_n), bias(s.OC), out(out_n), out_f(out_n), scratch;
        util::fill_uniform(in.data(), in_n, rng);
        util::fill_uniform(w.data(), w_n, rng, -0.1f, 0.1f);
        util::fill_uniform(bias.data(), s.OC, rng);
        const double sep = util::time_best([&] {
            conv::direct3x3(s, in.data(), w.data(), nullptr, false, out.data(), scratch);
            conv::add_bias(s.OC, hw, bias.data(), out.data());
            conv::relu(out_n, out.data());
        });
        const double fus = util::time_best(
            [&] { conv::direct3x3(s, in.data(), w.data(), bias.data(), true, out_f.data(), scratch); });
        const double ferr = util::max_abs_diff(out.data(), out_f.data(), out_n);
        if (ferr > 1e-4) std::printf("  fused/unfused mismatch %.2e\n", ferr);
        fusion.push_back({L.name, out_n * 4.0 / (1 << 20), sep, fus});
    }

    std::printf("\nconv + bias + ReLU: three passes vs one fused pass (direct NEON kernel)\n");
    std::printf("%-14s %10s %14s %12s %9s\n", "layer", "output MB", "separate ms", "fused ms", "speedup");
    for (const FusionRow& f : fusion) {
        std::printf("%-14s %10.2f %14.3f %12.3f %8.2fx\n", f.name.c_str(), f.out_mb, f.sep * 1e3, f.fused * 1e3,
                    f.sep / f.fused);
        if (fcsv) std::fprintf(fcsv, "%s,%.3f,%.5f,%.5f,%.3f\n", f.name.c_str(), f.out_mb, f.sep * 1e3, f.fused * 1e3, f.sep / f.fused);
    }
    if (csv) std::fclose(csv);
    if (fcsv) std::fclose(fcsv);
    return 0;
}
