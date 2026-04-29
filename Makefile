.PHONY: build check test benchmark run release install clean

build:
	@cargo build --release

check:
	@cargo check

test:
	@./test.sh

benchmark:
	@cargo run -- benchmark $(ARGS)


run:
	@cargo run -- $(ARGS)

release install:
	@./build_and_release.sh

clean:
	@cargo clean
	@echo "Cleaned."
