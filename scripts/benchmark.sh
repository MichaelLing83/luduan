#luduan benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 3,5,8,10 --whisper-no-context true,false --whisper-single-segment true,false --whisper-best-of 1,3 --silence-threshold 0.001,0.003 --ollama-model none,qwen3.5:0.8b,qwen3.5:2b,gemma3:1b,deepseek-r1:1.5b
cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 3,6,10 --whisper-no-context true,false --whisper-single-segment true,false --whisper-best-of 1,3 --silence-threshold 0.001,0.003 --ollama-model none,qwen3.5:0.8b

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 3
#001 accuracy=42.86% model=large-v3 chunk=3 language=fixture ollama=none output=target/benchmark/1777446058/run-001
#  hp_sv_01                         sv    OK        126 chars     72 edits   42.86%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 6
#001 accuracy=32.54% model=large-v3 chunk=6 language=fixture ollama=none output=target/benchmark/1777446151/run-001
#  hp_sv_01                         sv    OK        126 chars     85 edits   32.54%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 9
#001 accuracy=65.87% model=large-v3 chunk=9 language=fixture ollama=none output=target/benchmark/1777446207/run-001
#  hp_sv_01                         sv    OK        126 chars     43 edits   65.87%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 12
#001 accuracy=72.22% model=large-v3 chunk=12 language=fixture ollama=none output=target/benchmark/1777446251/run-001
#  hp_sv_01                         sv    OK        126 chars     35 edits   72.22%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 15
#001 accuracy=73.02% model=large-v3 chunk=15 language=fixture ollama=none output=target/benchmark/1777446299/run-001
#  hp_sv_01                         sv    OK        126 chars     34 edits   73.02%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 15 --whisper-no-context true --whisper-single-segment true --whisper-best-of 3
#001 accuracy=75.40% model=large-v3 chunk=15 language=fixture ollama=none no_context=true single_segment=true best_of=3 silence=0.003 output=target/benchmark/1777447225/run-001
#  hp_sv_01                         sv    OK        126 chars     31 edits   75.40%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 15 --whisper-no-context true --whisper-single-segment true --whisper-best-of 3 --language sv
#001 accuracy=75.40% model=large-v3 chunk=15 language=sv ollama=none no_context=true single_segment=true best_of=3 silence=0.003 output=target/benchmark/1777447365/run-001
#  hp_sv_01                         sv    OK        126 chars     31 edits   75.40%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 25 --whisper-no-context true --whisper-single-segment true --whisper-best-of 3 --language sv
#001 accuracy=76.19% model=large-v3 chunk=25 language=sv ollama=none no_context=true single_segment=true best_of=3 silence=0.003 output=target/benchmark/1777447537/run-001
#  hp_sv_01                         sv    OK        126 chars     30 edits   76.19%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 25 --whisper-no-context true --whisper-single-segment true,false --whisper-best-of 3 --language sv --silence-threshold 0.001,0.003,0.005 --context-file ./tests/fixtures/e2e/context.txt
#006 accuracy=86.51% model=large-v3 chunk=25 language=sv ollama=none no_context=true single_segment=false best_of=3 silence=0.005 output=target/benchmark/1777448146/run-006
#  hp_sv_01                         sv    OK        126 chars     17 edits   86.51%

# cargo run -- benchmark --fixtures ./tests/fixtures/e2e --model-name large-v3 --chunk-seconds 25 --whisper-no-context true --whisper-single-segment true --whisper-best-of 3 --language sv --silence-threshold 0.003 --context-file ./tests/fixtures/e2e/context.txt --keep-punctuation
#001 accuracy=86.92% model=large-v3 chunk=25 language=sv ollama=none no_context=true single_segment=true best_of=3 silence=0.003 output=target/benchmark/1777448497/run-001
#  hp_sv_01                         sv    OK        130 chars     17 edits   86.92%