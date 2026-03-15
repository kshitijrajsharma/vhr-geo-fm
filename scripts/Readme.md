# patch_geo_bench.py

This script augments the GEO-Bench2 statistics by including object detection datasets that are also very high resolution (VHR). It evaluates how model scores change when object detection tasks under 10m resolution are incorporated.

By default, the script evaluates the core high-resolution datasets. When run with the `--extra` flag, it adds additional object detection benchmarks (EverWatch and NZCattle) to the aggregated capability score. The output is a ranked Markdown table of all fully fine-tuned models. The logic follows the original repository.

## Usage

1. Clone the leaderboard repository:

   ```bash
   git clone https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard.git
   ```

2. Place `patch_geo_bench.py` in the root directory of the cloned repository.
3. Install dependencies and run:

   ```bash
   uv run python patch_geo_bench.py
   ```

## References

- [GEO-Bench-2-Leaderboard](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard)  
  > "A leaderboard for evaluating geospatial foundation models on a diverse set of tasks."  
- [uv documentation](https://github.com/astral-sh/uv)  
