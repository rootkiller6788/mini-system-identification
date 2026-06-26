CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm
AR = ar
LIB = build/librls.a
SRCS = $(wildcard src/*.c)
OBJS = $(patsubst src/%.c,build/%.o,$(SRCS))
TEST_SRC = tests/test_rls.c
TEST_BIN = build/test_rls
EXAMPLE_SRCS = $(wildcard examples/*.c)
EXAMPLE_BINS = $(patsubst examples/%.c,build/%,$(EXAMPLE_SRCS))

.PHONY: all builddir clean test examples demo

all: builddir $(LIB)

builddir:
	@mkdir -p build

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test: all
	$(CC) $(CFLAGS) -o $(TEST_BIN) $(TEST_SRC) -Lbuild -lrls $(LDFLAGS)
	./$(TEST_BIN)

examples: all $(EXAMPLE_BINS)
	@echo "All examples built."

build/example_%: examples/example_%.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -Lbuild -lrls $(LDFLAGS)

demo: examples
	@echo "=== Running example_fir_id ==="
	./build/example_fir_id
	@echo ""
	@echo "=== Running example_arx_id ==="
	./build/example_arx_id
	@echo ""
	@echo "=== Running example_ill_conditioned ==="
	./build/example_ill_conditioned

clean:
	rm -rf build
