// main_gui.cpp
#include <arm_neon.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <thread>
#include <atomic>
#include <algorithm>

#include "imgui.h"
#include "implot.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

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

struct Result {
    double n;
    double scalar_time;
    double neon_time;
    double speedup() const { return neon_time > 1e-9 ? scalar_time / neon_time : 0.0; }
};

std::vector<size_t> generate_sizes() {
    std::vector<size_t> sizes;
    for (int exp = 1; exp <= 7; ++exp) {
        size_t base = static_cast<size_t>(pow(10, exp));
        for (int i = 1; i <= 10; ++i)
            sizes.push_back(base * i);
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
        for (size_t i = 0; i < n; ++i) data[i] = dist(gen);

        auto t1 = std::chrono::high_resolution_clock::now();
        volatile int64_t r1 = process_array_scalar(data.data(), n);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto t3 = std::chrono::high_resolution_clock::now();
        volatile int64_t r2 = process_array_neon(data.data(), n);
        auto t4 = std::chrono::high_resolution_clock::now();

        (void)r1; (void)r2;

        results.push_back({
            (double)n,
            std::chrono::duration<double>(t2 - t1).count(),
            std::chrono::duration<double>(t4 - t3).count()
        });
        progress = float(idx + 1) / total;
    }
    running = false;
}

ImVec4 speedup_color(double sp) {
    if (sp >= 2.0) return {0.2f, 1.0f, 0.3f, 1.0f};
    if (sp >= 1.5) return {0.6f, 1.0f, 0.3f, 1.0f};
    if (sp >= 1.0) return {1.0f, 0.9f, 0.2f, 1.0f};
    return {1.0f, 0.4f, 0.4f, 1.0f};
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1400, 900, "NEON Benchmark", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImPlot::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    bool log_x = true, log_y = true, show_sp = true;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("NEON Performance Benchmark", nullptr, 
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {12, 8});
        
        if (!running && ImGui::Button("Start Benchmark", {160, 32})) {
            running = true; progress = 0.0f;
            std::thread(run_benchmark).detach();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear", {100, 32})) results.clear();
        ImGui::SameLine();
        ImGui::Checkbox("Log(N)", &log_x);
        ImGui::SameLine();
        ImGui::Checkbox("Log(Time)", &log_y);
        ImGui::SameLine();
        ImGui::Checkbox("Speedup", &show_sp);
        
        ImGui::PopStyleVar();

        if (running) {
            ImGui::Spacing();
            ImGui::Text("Processing: %.0f%%", progress * 100);
            ImGui::ProgressBar(progress, {-1, 20});
        }

        if (!results.empty() && !running) {
            ImGui::Spacing();
            ImGui::SeparatorText("Summary");
            
            double avg_sp = 0, min_sp = 1e9, max_sp = 0;
            int faster = 0;
            for (auto& r : results) {
                double sp = r.speedup();
                avg_sp += sp;
                min_sp = std::min(min_sp, sp);
                max_sp = std::max(max_sp, sp);
                if (sp > 1.0) faster++;
            }
            avg_sp /= results.size();
            
            ImGui::Columns(4, nullptr, false);
            ImGui::Text("Average"); ImGui::NextColumn();
            ImGui::Text("Min"); ImGui::NextColumn();
            ImGui::Text("Max"); ImGui::NextColumn();
            ImGui::Text("NEON > Scalar"); ImGui::NextColumn();
            ImGui::Separator();
            
            ImGui::TextColored({0.4f,0.8f,1.0f,1.0f}, "%.3fx", avg_sp); ImGui::NextColumn();
            ImGui::TextColored({0.4f,0.8f,1.0f,1.0f}, "%.3fx", min_sp); ImGui::NextColumn();
            ImGui::TextColored({0.4f,0.8f,1.0f,1.0f}, "%.3fx", max_sp); ImGui::NextColumn();
            ImGui::Text("%d / %zu", faster, results.size()); ImGui::NextColumn();
            ImGui::Columns(1);
        }

        if (!results.empty()) {
            ImGui::Spacing();
            ImGui::SeparatorText("Results");
            
            if (ImGui::BeginTable("tbl", 4, 
                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersOuter | 
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit, 
                    {-1, 220})) {
                
                ImGui::TableSetupColumn("N", ImGuiTableColumnFlags_WidthFixed, 90);
                ImGui::TableSetupColumn("Scalar (s)", ImGuiTableColumnFlags_WidthFixed, 110);
                ImGui::TableSetupColumn("NEON (s)", ImGuiTableColumnFlags_WidthFixed, 110);
                ImGui::TableSetupColumn("Ratio", ImGuiTableColumnFlags_WidthFixed, 90);
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableHeadersRow();

                ImGuiListClipper clipper;
                clipper.Begin(results.size());
                while (clipper.Step()) {
                    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                        const auto& r = results[row];
                        ImGui::TableNextRow();
                        
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%.0f", r.n);
                        
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%.6f", r.scalar_time);
                        
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%.6f", r.neon_time);
                        
                        ImGui::TableSetColumnIndex(3);
                        double sp = r.speedup();
                        ImGui::TextColored(speedup_color(sp), "%.3fx", sp);
                    }
                }
                ImGui::EndTable();
            }
        }

        if (!results.empty() && ImGui::CollapsingHeader("Execution Time", 
                ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImPlot::BeginPlot("##time", {-1, 280})) {
                ImPlot::SetupAxes("Array Size (N)", "Time (seconds)");
                if (log_x) ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                if (log_y) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0, 0);
                
                std::vector<double> x, y1, y2;
                for (auto& r : results) {
                    x.push_back(r.n);
                    y1.push_back(r.scalar_time);
                    y2.push_back(r.neon_time);
                }
                
                ImPlot::PlotLine("Scalar", x.data(), y1.data(), x.size());
                ImPlot::PlotLine("NEON", x.data(), y2.data(), x.size());
                ImPlot::EndPlot();
            }
        }

        if (!results.empty() && show_sp && ImGui::CollapsingHeader("Speedup Ratio", 
                ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImPlot::BeginPlot("##sp", {-1, 200})) {
                ImPlot::SetupAxes("Array Size (N)", "Speedup (Scalar / NEON)");
                if (log_x) ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                
                std::vector<double> ref_x = {results.front().n, results.back().n};
                std::vector<double> ref_y = {1.0, 1.0};
                ImPlot::PlotLine("##ref", ref_x.data(), ref_y.data(), 2, 
                    ImPlotLineFlags_Shaded, 0, 0, {0.3f,0.3f,0.3f,0.15f});
                
                std::vector<double> x, y;
                for (auto& r : results) {
                    x.push_back(r.n);
                    y.push_back(r.speedup());
                }
                ImPlot::PlotLine("Speedup", x.data(), y.data(), x.size(), 
                    ImPlotLineFlags_Shaded, 0, 0, {0.3f,0.7f,1.0f,0.2f});
                
                ImPlot::EndPlot();
            }
        }

        ImGui::End();
        ImGui::Render();
        
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
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
