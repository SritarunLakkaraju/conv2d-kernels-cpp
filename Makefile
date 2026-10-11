include deps/flags.mk

all: bench_conv

bench_conv: bench_conv.cpp conv.h deps/gemm.h deps/util.h
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $< -o $@ $(OMP_LDFLAGS)

run: bench_conv
	mkdir -p results
	./bench_conv results | tee results/conv.txt

clean:
	rm -f bench_conv

.PHONY: all run clean
