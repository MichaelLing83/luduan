.PHONY: build check test run release install clean

build:
	@cargo build --release

check:
	@cargo check

test:
	@./test.sh

run:
	@cargo run -- $(ARGS)

release install:
	@./build_and_release.sh

clean:
	@cargo clean
	@echo "Cleaned."
