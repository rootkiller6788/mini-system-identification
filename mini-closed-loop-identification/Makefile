CC = gcc
CFLAGS = -Wall -Wextra -Wno-misleading-indentation -Wno-unused-parameter -Wno-unused-variable -std=c11 -O2 -Iinclude
LDFLAGS = -lm

SRCS = src/clid_types.c src/clid_direct.c src/clid_indirect.c        src/clid_joint_io.c src/clid_iv.c src/clid_subspace.c        src/clid_youla.c src/clid_validation.c

OBJS = $(SRCS:.c=.o)
TARGET = libclid.a
TEST_TARGET = test_clid

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^

src/%.o: src/%.c include/*.h
	$(CC) $(CFLAGS) -c $< -o $@

test: $(SRCS) tests/test_clid.c
	$(CC) $(CFLAGS) tests/test_clid.c $(SRCS) $(LDFLAGS) -o $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -f src/*.o $(TARGET) $(TEST_TARGET) *.exe

examples: $(SRCS)
	for f in examples/*.c; do 		out=$${f%.c}; 		$(CC) $(CFLAGS) $$f $(SRCS) $(LDFLAGS) -o $$out; 	done
