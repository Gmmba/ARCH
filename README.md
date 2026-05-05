sudo apt install -y libsdl2-dev libgl-dev cmake build-essential  

git clone https://github.com/ocornut/imgui.git  

git clone https://github.com/epezent/implot.git  

cmake -B build && cmake --build build -j4
./build/neon_gui  


aarch64-linux-gnu-g++ -O3 -march=armv8-a+simd -o neon_array neon_array.cpp  

arm-linux-gnueabihf-g++ -O3 -mfpu=neon -mfloat-abi=hard -o neon_array neon_array.cpp  


g++ -O3 -march=armv8-a -DARM_NEON_AVAILABLE \\
    neon_implot_gui.cpp \\
    imgui/imgui.cpp imgui/imgui_draw.cpp \\
    imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \\
    imgui/backends/imgui_impl_sdl2.cpp \\
    imgui/backends/imgui_impl_opengl3.cpp \\
    implot/implot.cpp implot/implot_items.cpp \\
    -I imgui -I imgui/backends -I implot \\
    $(sdl2-config --cflags --libs) -lGL \\
    -o neon_gui  
    

g++ -O3 \\
    neon_implot_gui.cpp \\
    imgui/imgui.cpp imgui/imgui_draw.cpp \\
    imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \\
    imgui/backends/imgui_impl_sdl2.cpp \\
    imgui/backends/imgui_impl_opengl3.cpp \\
    implot/implot.cpp implot/implot_items.cpp \\
    -I imgui -I imgui/backends -I implot \\
    $(sdl2-config --cflags --libs) -lGL \\
    -o neon_gui  
    
