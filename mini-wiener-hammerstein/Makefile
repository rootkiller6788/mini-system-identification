CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm
LIB = build/libwh.a
SRCS = $(wildcard src/*.c)
OBJS = $(patsubst src/%.c,build/%.o,$(SRCS))
TEST_SRC = tests/test_wh_model.c
TEST_BIN = build/test_wh_model
EXAMPLE_SRCS = $(wildcard examples/*.c)
EXAMPLE_BINS = $(patsubst examples/%.c,build/%,$(EXAMPLE_SRCS))
BENCH_SRC = benches/bench_core.c
BENCH_BIN = build/bench_core

.PHONY: all builddir clean test examples demo bench

all: builddir $(LIB)

builddir:
	@mkdir -p build

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test: all
	$(CC) $(CFLAGS) -o $(TEST_BIN) $(TEST_SRC) -Lbuild -lwh $(LDFLAGS)
	./$(TEST_BIN)

examples: all $(EXAMPLE_BINS)
	@echo "All examples built."

build/example_%: examples/example_%.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -Lbuild -lwh $(LDFLAGS)

build/bench_%: benches/bench_%.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -Lbuild -lwh $(LDFLAGS)

demo: examples
	@echo "=== Running Wiener-Hammerstein Benchmark ==="
	./build/example_wh_benchmark
	@echo ""
	@echo "=== Running Industrial Actuator Example ==="
	./build/example_wh_industrial
	@echo ""
	@echo "=== Running Biological Dose-Response Example ==="
	./build/example_wh_bio

bench: all
	$(CC) $(CFLAGS) -o $(BENCH_BIN) $(BENCH_SRC) -Lbuild -lwh $(LDFLAGS)
	./$(BENCH_BIN)

clean:
	rm -rf build
