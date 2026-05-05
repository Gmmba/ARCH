/*
 *
 *   g++ -O3 -march=armv8-a -o neon_gui neon_implot_gui.cpp \
 *       $(sdl2-config --cflags --libs) \
 *       -lGL -limgui -limplot \
 *       -DARM_NEON_AVAILABLE
 *
 *   g++ -O3 -o neon_gui neon_implot_gui.cpp \
 *       $(sdl2-config --cflags --libs) \
 *       -lGL -limgui -limplot
 *
 *   -I./imgui -I./implot imgui/*.cpp implot/implot*.cpp \
 *   imgui/backends/imgui_impl_sdl2.cpp \
 *   imgui/backends/imgui_impl_opengl3.cpp
 * 
 *   sudo apt install libsdl2-dev libgl-dev
 *   git clone https://github.com/ocornut/imgui
 *   git clone https://github.com/epezent/implot
 */

#ifdef ARM_NEON_AVAILABLE
#  include <arm_neon.h>
#endif

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>

__attribute__((noinline, optimize("no-tree-vectorize")))
static int64_t process_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t v = data[i];
        if      (v > 0) sum += v;
        else if (v < 0) sum -= v;
    }
    return sum;
}

#ifdef ARM_NEON_AVAILABLE

__attribute__((noinline))
static int64_t process_neon(const int32_t* __restrict__ data, size_t n) {
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    size_t i = 0;

    for (; i + 7 < n; i += 8) {
        __builtin_prefetch(data + i + 32, 0, 1);

        int32x4_t v0 = vld1q_s32(data + i);
        int32x4_t v1 = vld1q_s32(data + i + 4);

        int32x4_t s0 = vshrq_n_s32(v0, 31);
        int32x4_t s1 = vshrq_n_s32(v1, 31);

        int32x4_t a0 = vsubq_s32(veorq_s32(v0, s0), s0);
        int32x4_t a1 = vsubq_s32(veorq_s32(v1, s1), s1);

        uint32x4_t nz0 = vcgtq_s32(a0, vdupq_n_s32(0));
        uint32x4_t nz1 = vcgtq_s32(a1, vdupq_n_s32(0));
        a0 = vandq_s32(a0, vreinterpretq_s32_u32(nz0));
        a1 = vandq_s32(a1, vreinterpretq_s32_u32(nz1));

        acc0 = vaddq_s64(acc0, vpaddlq_s32(a0));
        acc1 = vaddq_s64(acc1, vpaddlq_s32(a1));
    }

    for (; i + 3 < n; i += 4) {
        int32x4_t v   = vld1q_s32(data + i);
        int32x4_t s   = vshrq_n_s32(v, 31);
        int32x4_t a   = vsubq_s32(veorq_s32(v, s), s);
        uint32x4_t nz = vcgtq_s32(a, vdupq_n_s32(0));
        a = vandq_s32(a, vreinterpretq_s32_u32(nz));
        acc0 = vaddq_s64(acc0, vpaddlq_s32(a));
    }

    int64x2_t acc = vaddq_s64(acc0, acc1);
    int64_t sum = vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1);

    for (; i < n; ++i) {
        int32_t v = data[i];
        if      (v > 0) sum += v;
        else if (v < 0) sum -= v;
    }
    return sum;
}

#else

__attribute__((noinline))
static int64_t process_neon(const int32_t* __restrict__ data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t v = data[i];
        if      (v > 0) sum += v;
        else if (v < 0) sum -= v;
    }
    return sum;
}

#endif

using Fn = int64_t(*)(const int32_t*, size_t);

static double run_bench(Fn fn, const int32_t* data, size_t n, int iters) {
    volatile int64_t warmup = fn(data, n);
    (void)warmup;
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile int64_t r = 0;
    for (int i = 0; i < iters; ++i) r = fn(data, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)r;
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

struct AppState {
    int  array_size  = 100000;
    int  iterations  = 100;
    int  range_min   = -1000;
    int  range_max   =  1000;
    int  seed        = 42;
    bool use_aligned = true;

    double scalar_ms = 0.0;
    double neon_ms   = 0.0;
    double speedup   = 0.0;
    int64_t scalar_result = 0;
    int64_t neon_result   = 0;
    bool    results_match = true;
    bool    ran           = false;

    std::vector<double> history_scalar;
    std::vector<double> history_neon;
    std::vector<double> history_speedup;
    std::vector<double> history_x;

    std::vector<double> sweep_sizes;
    std::vector<double> sweep_scalar_ms;
    std::vector<double> sweep_neon_ms;
    std::vector<double> sweep_speedup;
    bool sweep_done = false;

    std::vector<double> hist_bins;
    std::vector<double> hist_counts;
    bool hist_done = false;

    std::vector<int32_t> last_data;
};

static std::vector<int32_t> make_data(int n, int lo, int hi, int seed, bool aligned) {
    std::vector<int32_t> v(n + (aligned ? 4 : 0));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(lo, hi);
    for (int i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

static void build_histogram(AppState& s) {
    const int BINS = 40;
    s.hist_bins.resize(BINS);
    s.hist_counts.resize(BINS, 0.0);

    int lo = s.range_min, hi = s.range_max;
    double width = double(hi - lo) / BINS;
    for (int b = 0; b < BINS; ++b)
        s.hist_bins[b] = lo + b * width + width * 0.5;

    for (int32_t v : s.last_data) {
        int b = int((v - lo) / width);
        b = std::clamp(b, 0, BINS - 1);
        s.hist_counts[b]++;
    }
    s.hist_done = true;
}

static void run_sweep(AppState& s) {
    s.sweep_sizes.clear();
    s.sweep_scalar_ms.clear();
    s.sweep_neon_ms.clear();
    s.sweep_speedup.clear();

    const std::vector<int> sizes = {
        1000, 2000, 5000, 10000, 25000, 50000,
        100000, 250000, 500000, 1000000
    };

    for (int sz : sizes) {
        auto data = make_data(sz, s.range_min, s.range_max, s.seed, s.use_aligned);
        double sm = run_bench(process_scalar, data.data(), sz, 20);
        double nm = run_bench(process_neon,   data.data(), sz, 20);
        s.sweep_sizes.push_back(double(sz));
        s.sweep_scalar_ms.push_back(sm);
        s.sweep_neon_ms.push_back(nm);
        s.sweep_speedup.push_back(nm > 0.0 ? sm / nm : 0.0);
    }
    s.sweep_done = true;
}

static void run_bench_once(AppState& s) {
    auto data = make_data(s.array_size, s.range_min, s.range_max, s.seed, s.use_aligned);
    s.last_data.assign(data.begin(), data.begin() + s.array_size);
    build_histogram(s);

    s.scalar_result = process_scalar(data.data(), s.array_size);
    s.neon_result   = process_neon  (data.data(), s.array_size);
    s.results_match = (s.scalar_result == s.neon_result);

    s.scalar_ms = run_bench(process_scalar, data.data(), s.array_size, s.iterations);
    s.neon_ms   = run_bench(process_neon,   data.data(), s.array_size, s.iterations);
    s.speedup   = (s.neon_ms > 0.0) ? s.scalar_ms / s.neon_ms : 0.0;

    double run_id = double(s.history_x.empty() ? 1 : s.history_x.back() + 1);
    s.history_x.push_back(run_id);
    s.history_scalar.push_back(s.scalar_ms);
    s.history_neon.push_back(s.neon_ms);
    s.history_speedup.push_back(s.speedup);

    s.ran = true;
}

static void render_gui(AppState& s) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::Begin("ARM NEON Benchmark", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove       |
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::BeginChild("Settings", ImVec2(280, 0), true);

    ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Параметры");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SliderInt("Размер массива", &s.array_size, 1000, 1000000);
    ImGui::SliderInt("Итерации", &s.iterations, 10, 500);
    ImGui::SliderInt("Min значение", &s.range_min, -100000, 0);
    ImGui::SliderInt("Max значение", &s.range_max, 0, 100000);
    ImGui::SliderInt("Seed", &s.seed, 0, 9999);
    ImGui::Checkbox("Выровнять память (16 байт)", &s.use_aligned);
    ImGui::Spacing();

    if (ImGui::Button("Запустить бенчмарк", ImVec2(-1, 36)))
        run_bench_once(s);

    ImGui::Spacing();
    if (ImGui::Button("Sweep по размерам", ImVec2(-1, 36)))
        run_sweep(s);

    if (s.history_x.size() > 1) {
        ImGui::Spacing();
        if (ImGui::Button("Очистить историю", ImVec2(-1, 28))) {
            s.history_x.clear();
            s.history_scalar.clear();
            s.history_neon.clear();
            s.history_speedup.clear();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Платформа");
#ifdef ARM_NEON_AVAILABLE
    ImGui::TextColored(ImVec4(0.3f,1.0f,0.3f,1.0f), "ARM NEON: АКТИВЕН");
#else
    ImGui::TextColored(ImVec4(1.0f,0.7f,0.2f,1.0f), "ARM NEON: x86 fallback");
#endif

    if (s.ran) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Результаты");
        ImGui::Spacing();

        ImGui::Text("Scalar: %.4f мс", s.scalar_ms);
        ImGui::Text("NEON:   %.4f мс", s.neon_ms);

        ImGui::Spacing();
        float ratio = float(s.speedup);
        ImVec4 color = ratio >= 3.0f
            ? ImVec4(0.2f, 1.0f, 0.3f, 1.0f)
            : ImVec4(1.0f, 0.6f, 0.1f, 1.0f);
        ImGui::TextColored(color, "Ускорение: %.2fx", s.speedup);
        ImGui::ProgressBar(std::min(ratio / 6.0f, 1.0f), ImVec2(-1, 0),
            (std::to_string((int)(ratio * 10) / 10.0) + "x").c_str());

        ImGui::Spacing();
        if (s.results_match)
            ImGui::TextColored(ImVec4(0.2f,1.0f,0.2f,1.0f), "Результаты совпадают");
        else
            ImGui::TextColored(ImVec4(1.0f,0.2f,0.2f,1.0f), "ОШИБКА: расхождение!");

        ImGui::Text("Сумма: %lld", (long long)s.scalar_result);
    }

    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("Plots", ImVec2(0, 0), false);

    float plot_w  = ImGui::GetContentRegionAvail().x;
    float plot_h1 = 220.0f;
    float plot_h2 = 200.0f;

    if (s.hist_done && ImPlot::BeginPlot("Распределение входных данных",
            ImVec2(plot_w * 0.5f - 4, plot_h1))) {
        ImPlot::SetupAxes("Значение", "Частота");
        double bw = (s.range_max - s.range_min) / double(s.hist_bins.size());
        ImPlot::SetNextFillStyle(ImVec4(0.3f,0.6f,1.0f,0.8f));
        ImPlot::PlotBars("Элементы",
            s.hist_bins.data(), s.hist_counts.data(),
            (int)s.hist_bins.size(), bw * 0.9);
        ImPlot::EndPlot();
    }

    if (s.ran) {
        ImGui::SameLine();
        if (ImPlot::BeginPlot("Время выполнения (мс)",
                ImVec2(plot_w * 0.5f - 4, plot_h1))) {
            ImPlot::SetupAxes("Версия", "мс");
            const char* labels[] = {"Scalar", "NEON"};
            double vals[]   = {s.scalar_ms, s.neon_ms};
            ImPlot::SetNextFillStyle(ImVec4(0.9f, 0.4f, 0.3f, 1.0f));
            ImPlot::PlotBars("Scalar", &vals[0], 1, 0.4, 0);
            ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.9f, 0.5f, 1.0f));
            ImPlot::PlotBars("NEON",   &vals[1], 1, 0.4, 1);
            (void)labels;
            ImPlot::EndPlot();
        }
    }

    if (s.history_x.size() > 1) {
        if (ImPlot::BeginPlot("История ускорения (×)",
                ImVec2(plot_w, plot_h2))) {
            ImPlot::SetupAxes("Прогон №", "Ускорение (×)");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 8.0, ImPlotCond_Once);
            double xs3[] = {s.history_x.front(), s.history_x.back()};
            double ys3[] = {3.0, 3.0};
            ImPlot::SetNextLineStyle(ImVec4(1,1,0,0.6f), 1.5f);
            ImPlot::PlotLine("Цель 3×", xs3, ys3, 2);

            ImPlot::SetNextLineStyle(ImVec4(0.3f,1.0f,0.5f,1.0f), 2.0f);
            ImPlot::PlotLine("Ускорение",
                s.history_x.data(),
                s.history_speedup.data(),
                (int)s.history_x.size());
            ImPlot::SetNextFillStyle(ImVec4(0.3f,1.0f,0.5f,0.15f));
            ImPlot::PlotShaded("##shade",
                s.history_x.data(),
                s.history_speedup.data(),
                (int)s.history_x.size(), 0.0);
            ImPlot::EndPlot();
        }

        if (ImPlot::BeginPlot("История времени выполнения (мс)",
                ImVec2(plot_w, plot_h2))) {
            ImPlot::SetupAxes("Прогон №", "мс");
            ImPlot::SetNextLineStyle(ImVec4(0.9f,0.4f,0.3f,1.0f), 2.0f);
            ImPlot::PlotLine("Scalar",
                s.history_x.data(), s.history_scalar.data(),
                (int)s.history_x.size());
            ImPlot::SetNextLineStyle(ImVec4(0.3f,0.6f,1.0f,1.0f), 2.0f);
            ImPlot::PlotLine("NEON",
                s.history_x.data(), s.history_neon.data(),
                (int)s.history_x.size());
            ImPlot::EndPlot();
        }
    }

    if (s.sweep_done) {
        if (ImPlot::BeginPlot("Sweep: время vs. размер массива",
                ImVec2(plot_w * 0.5f - 4, plot_h2 + 20))) {
            ImPlot::SetupAxes("Размер (элементов)", "мс");
            ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
            ImPlot::SetNextLineStyle(ImVec4(0.9f,0.4f,0.3f,1.0f), 2.0f);
            ImPlot::PlotLine("Scalar",
                s.sweep_sizes.data(), s.sweep_scalar_ms.data(),
                (int)s.sweep_sizes.size());
            ImPlot::SetNextLineStyle(ImVec4(0.3f,0.9f,0.5f,1.0f), 2.0f);
            ImPlot::PlotLine("NEON",
                s.sweep_sizes.data(), s.sweep_neon_ms.data(),
                (int)s.sweep_sizes.size());
            ImPlot::EndPlot();
        }

        ImGui::SameLine();
        if (ImPlot::BeginPlot("Sweep: ускорение vs. размер",
                ImVec2(plot_w * 0.5f - 4, plot_h2 + 20))) {
            ImPlot::SetupAxes("Размер (элементов)", "Ускорение (×)");
            ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
            ImPlot::SetNextFillStyle(ImVec4(0.3f,1.0f,0.5f,0.2f));
            ImPlot::PlotShaded("##shade",
                s.sweep_sizes.data(), s.sweep_speedup.data(),
                (int)s.sweep_sizes.size(), 0.0);
            ImPlot::SetNextLineStyle(ImVec4(0.3f,1.0f,0.5f,1.0f), 2.5f);
            ImPlot::PlotLine("Ускорение NEON",
                s.sweep_sizes.data(), s.sweep_speedup.data(),
                (int)s.sweep_sizes.size());
            double xs3[] = {s.sweep_sizes.front(), s.sweep_sizes.back()};
            double ys3[] = {3.0, 3.0};
            ImPlot::SetNextLineStyle(ImVec4(1,1,0,0.7f), 1.5f);
            ImPlot::PlotLine("Цель 3×", xs3, ys3, 2);
            ImPlot::EndPlot();
        }
    }

    ImGui::EndChild();
    ImGui::End();
}

int main(int /*argc*/, char** /*argv*/) {

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        SDL_Log("SDL_Init error: %s", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    SDL_Window* window = SDL_CreateWindow(
        "ARM NEON Benchmark — ImPlot GUI",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1400, 820,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) {
        SDL_Log("SDL_CreateWindow error: %s", SDL_GetError());
        return 1;
    }

    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_ctx);
    SDL_GL_SetSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding  = 6.0f;
    style.FrameRounding   = 4.0f;
    style.GrabRounding    = 4.0f;
    style.ChildRounding   = 4.0f;

    ImGui_ImplSDL2_InitForOpenGL(window, gl_ctx);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    AppState state;

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                running = false;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window))
                running = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        render_gui(state);

        ImGui::Render();
        int w, h;
        SDL_GetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.10f, 0.10f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}