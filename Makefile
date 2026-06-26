# Makefile — Module 10: mini-system-identification
# Batch build and test for all 8 submodules

SUBMODULES = mini-closed-loop-identification \
             mini-frequency-domain-id \
             mini-nonlinear-system-id \
             mini-prediction-error-method \
             mini-regularized-least-squares \
             mini-subspace-identification \
             mini-uncertainty-quantification \
             mini-wiener-hammerstein

.PHONY: all test clean help

help:
	@echo "Module 10: System Identification"
	@echo "  make test    — Run tests for all 8 submodules"
	@echo "  make clean   — Clean all submodule build artifacts"
	@echo "  make list    — List submodules with line counts"

test:
	@passed=0; failed=0; \
	for d in $(SUBMODULES); do \
		echo "=== [$$d] ==="; \
		if (cd $$d && $(MAKE) test > /dev/null 2>&1); then \
			echo "  PASS: $$d"; passed=$$((passed+1)); \
		else \
			echo "  FAIL: $$d"; failed=$$((failed+1)); \
		fi; \
	done; \
	echo ""; \
	echo "=========================================="; \
	echo "Results: $$passed passed, $$failed failed"; \
	[ $$failed -eq 0 ] && echo "All tests passed!" || echo "Some tests FAILED."

clean:
	@for d in $(SUBMODULES); do \
		(cd $$d && $(MAKE) clean 2>/dev/null) || true; \
	done
	@echo "Cleaned all submodules."

list:
	@for d in $(SUBMODULES); do \
		lines=$$(find $$d/include $$d/src -name "*.c" -o -name "*.h" 2>/dev/null | xargs cat 2>/dev/null | wc -l); \
		printf "  %-40s %6d lines\n" "$$d" $$lines; \
	done
