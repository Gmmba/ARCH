#include <arm_neon.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <thread>
#include <atomic>

#include "imgui.h"
#include "implot.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

// ======================= ЯДРО =======================

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    return sum;
}

int64_t process_array_neon(const int32_t* data, size_t n) {
    int64_t sum = 0;

    int32x4_t acc = vdupq_n_s32(0);
    int32x4_t zero = vdupq_n_s32(0);

    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __builtin_prefetch(data + i + 16);

        int32x4_t vec = vld1q_s32(data + i);

        uint32x4_t mask_pos = vcgtq_s32(vec, zero);
        uint32x4_t mask_neg = vcltq_s32(vec, zero);

        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);

        int32x4_t pos_part = vbslq_s32(mask_pos, vec, zero);
        int32x4_t neg_part = vbslq_s32(mask_neg, abs_val, zero);

        int32x4_t contrib = vorrq_s32(pos_part, neg_part);

        acc = vaddq_s32(acc, contrib);
    }

    int32_t temp[4];
    vst1q_s32(temp, acc);

    sum += temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }

    return sum;
}

// ======================= BENCHMARK =======================

struct Result {
    double n;
    double scalar_time;
    double neon_time;
};

std::vector<size_t> generate_sizes() {
    std::vector<size_t> sizes;
    for (int exp = 1; exp <= 7; ++exp) {
        size_t base = pow(10, exp);
        for (int i = 1; i <= 10; ++i) {
            sizes.push_back(base * i);
        }
    }
    return sizes;
}

std::vector<Result> results;
std::atomic<bool> running(false);
std::atomic<float> progress(0.0f);

void run_benchmark() {
    results.clear();
    auto sizes = generate_sizes();

    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);

    size_t total = sizes.size();

    for (size_t idx = 0; idx < total; ++idx) {
        size_t n = sizes[idx];

        std::vector<int32_t> data(n);
        for (size_t i = 0; i < n; ++i)
            data[i] = dist(gen);

        auto t1 = std::chrono::high_resolution_clock::now();
        process_array_scalar(data.data(), n);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto t3 = std::chrono::high_resolution_clock::now();
        process_array_neon(data.data(), n);
        auto t4 = std::chrono::high_resolution_clock::now();

        double scalar_sec = std::chrono::duration<double>(t2 - t1).count();
        double neon_sec   = std::chrono::duration<double>(t4 - t3).count();

        results.push_back({(double)n, scalar_sec, neon_sec});

        progress = float(idx + 1) / total;
    }

    running = false;
}

// ======================= MAIN GUI =======================

int main() {
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1280, 720, "NEON Benchmark", NULL, NULL);
    glfwMakeContextCurrent(window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("NEON Benchmark");

        if (!running && ImGui::Button("Run Benchmark")) {
            running = true;
            progress = 0.0f;
            std::thread(run_benchmark).detach();
        }

        if (running) {
            ImGui::Text("Running...");
            ImGui::ProgressBar(progress, ImVec2(-1, 0));
        }

        if (!results.empty() && !running) {

            // ===== ТАБЛИЦА =====
            if (ImGui::BeginTable("table", 4)) {
                ImGui::TableSetupColumn("N");
                ImGui::TableSetupColumn("Scalar (s)");
                ImGui::TableSetupColumn("NEON (s)");
                ImGui::TableSetupColumn("Speedup");
                ImGui::TableHeadersRow();

                for (auto& r : results) {
                    ImGui::TableNextRow();

                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%.0f", r.n);

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.6f", r.scalar_time);

                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.6f", r.neon_time);

                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.2fx", r.scalar_time / r.neon_time);
                }

                ImGui::EndTable();
            }

            // ===== ГРАФИК =====
            if (ImPlot::BeginPlot("Performance")) {

                std::vector<double> x, y1, y2;

                for (auto& r : results) {
                    x.push_back(r.n);
                    y1.push_back(r.scalar_time);
                    y2.push_back(r.neon_time);
                }

                ImPlot::SetupAxes("N", "Time (seconds)");
                ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

                ImPlot::PlotLine("Scalar", x.data(), y1.data(), x.size());
                ImPlot::PlotLine("NEON", x.data(), y2.data(), x.size());

                ImPlot::EndPlot();
            }
        }

        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}