P=/mnt/storage/s43-prima-baseline-20261016
export PATH=/mnt/storage/s21_deps/cuda-13.2.1/bin:$PATH CUDA_PATH=/mnt/storage/s21_deps/cuda-13.2.1
cd $P/prima.cpp && make -j 12 GGML_CUDA=1 USE_HIGHS=1 GGML_NO_CCACHE=1 CUDA_DOCKER_ARCH=sm_89 CUDA_PATH=/mnt/storage/s21_deps/cuda-13.2.1 CPPFLAGS="-I$P/deps/include -isystem $P/deps/include/highs" LDFLAGS="-L$P/deps/lib -lpthread" llama-server llama-cli > $P/logs/prima-cuda-build.log 2>&1; echo "PRIMA_BUILD_EXIT=$?" >> $P/logs/prima-cuda-build.log
