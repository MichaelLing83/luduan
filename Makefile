.PHONY: build check run release install clean

build:
	@cargo build --release

check:
	@cargo check

run:
	@cargo run -- $(ARGS)

release install:
	@./build_and_release.sh

clean:
	@cargo clean
	@echo "Cleaned."
