from __future__ import annotations

import argparse
import ast
import html
import math
import os
import re
import runpy
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass


PROJECT_DIR = Path(__file__).resolve().parent
QUERY_SOURCE = PROJECT_DIR / "spotify_query.py"
ML_SOURCE = PROJECT_DIR / "spotify_ML1.py"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "dashboard_output"
DEFAULT_HDFS_PATH = "hdfs://localhost:9000/spark_data/SpotifyFeatures.csv"
DEFAULT_SPARK_MEMORY = "1g"
DEFAULT_SHUFFLE_PARTITIONS = 8
DEFAULT_QUERY_LIMIT = 4


SPOTIFY_SCHEMA = StructType(
    [
        StructField("genre", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("track_name", StringType(), True),
        StructField("track_id", StringType(), True),
        StructField("popularity", IntegerType(), True),
        StructField("acousticness", DoubleType(), True),
        StructField("danceability", DoubleType(), True),
        StructField("duration_ms", IntegerType(), True),
        StructField("energy", DoubleType(), True),
        StructField("instrumentalness", DoubleType(), True),
        StructField("key", StringType(), True),
        StructField("liveness", DoubleType(), True),
        StructField("loudness", DoubleType(), True),
        StructField("mode", StringType(), True),
        StructField("speechiness", DoubleType(), True),
        StructField("tempo", DoubleType(), True),
        StructField("time_signature", StringType(), True),
        StructField("valence", DoubleType(), True),
    ]
)


@dataclass
class QuerySpec:
    index: int
    title: str
    sql: str
    source_line: int


@dataclass
class QueryResult:
    spec: QuerySpec
    frame: pd.DataFrame | None = None
    error: str | None = None


@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def clean_title(value: str, fallback: str) -> str:
    text = " ".join(str(value).replace("\n", " ").split())
    text = text.replace("â€“", "-").replace("–", "-").strip(" -:")
    return text[:120] if text else fallback


def short_text(value: Any, max_len: int = 28) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def extract_hdfs_path(query_path: Path) -> str:
    if not query_path.exists():
        return DEFAULT_HDFS_PATH
    source = query_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\.load\(\s*['\"]([^'\"]+SpotifyFeatures\.csv)['\"]\s*\)", source)
    return match.group(1) if match else DEFAULT_HDFS_PATH


def collect_query_labels(tree: ast.Module) -> list[tuple[int, str]]:
    labels: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "print":
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "[Q" in arg.value:
            labels.append((node.lineno, clean_title(arg.value, f"Query at line {node.lineno}")))
    return sorted(labels, key=lambda item: item[0])


def nearest_label(labels: list[tuple[int, str]], line_no: int, fallback: str) -> str:
    previous = [label for label_line, label in labels if label_line < line_no]
    return previous[-1] if previous else fallback


def is_spark_sql_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sql"
        and bool(node.args)
    )


def render_sql_expression(expr: ast.AST, query_path: Path) -> list[tuple[str, str]]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [("", expr.value)]

    if isinstance(expr, ast.JoinedStr):
        compiled = compile(ast.Expression(expr), str(query_path), "eval")
        contexts = [
            {"genre": "Pop", "threshold": 77},
            {"genre": "Rap", "threshold": 73},
        ]
        return [
            (
                f"{context['genre']} threshold {context['threshold']}",
                str(eval(compiled, {}, context)),
            )
            for context in contexts
        ]

    return []


def extract_query_specs(query_path: Path, query_limit: int) -> list[QuerySpec]:
    source = query_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(query_path))
    labels = collect_query_labels(tree)

    specs: list[QuerySpec] = []
    sql_nodes = sorted(
        [node for node in ast.walk(tree) if is_spark_sql_call(node)],
        key=lambda node: node.lineno,
    )
    for node in sql_nodes:
        title = nearest_label(labels, node.lineno, f"Query {len(specs) + 1}")
        for suffix, sql in render_sql_expression(node.args[0], query_path):
            full_title = f"{title} - {suffix}" if suffix else title
            specs.append(
                QuerySpec(
                    index=len(specs) + 1,
                    title=clean_title(full_title, f"Query {len(specs) + 1}"),
                    sql=sql,
                    source_line=node.lineno,
                )
            )
            if len(specs) >= query_limit:
                return specs

    return specs


def configure_spark_environment(memory: str):
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", f"--driver-memory {memory} pyspark-shell")


def build_spark_session(app_name: str, memory: str, shuffle_partitions: int) -> SparkSession:
    configure_spark_environment(memory)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", "2")
        .config("spark.driver.maxResultSize", "512m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def stop_spark_safely(spark: SparkSession | None):
    if spark is None:
        return
    try:
        spark.stop()
    except Exception as exc:
        print(f"[WARN] Spark stop skipped: {type(exc).__name__}: {exc}")


def load_tracks_view(spark: SparkSession, hdfs_path: str):
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("quote", '"')
        .option("escape", '"')
        .schema(SPOTIFY_SCHEMA)
        .load(hdfs_path)
    )

    if "track_id" in df.columns:
        window_spec = Window.partitionBy("track_id").orderBy(col("genre").asc())
        df = df.withColumn("rn", row_number().over(window_spec)).filter(col("rn") == 1).drop("rn")

    df.createOrReplaceTempView("tracks_deduplicated")
    return df


def execute_query_specs(spark: SparkSession, specs: list[QuerySpec]) -> list[QueryResult]:
    results: list[QueryResult] = []
    for spec in specs:
        try:
            frame = spark.sql(spec.sql).toPandas()
            results.append(QueryResult(spec=spec, frame=frame))
            print(f"[OK] Query {spec.index}: {spec.title} ({len(frame)} rows)")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            results.append(QueryResult(spec=spec, error=message))
            print(f"[ERROR] Query {spec.index}: {spec.title} -> {message}")
    return results


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name not in numeric_columns(frame)]


def plot_table(ax: plt.Axes, frame: pd.DataFrame, title: str):
    ax.axis("off")
    preview = frame.head(8).copy()
    for column in preview.columns:
        preview[column] = preview[column].map(lambda value: short_text(value, 20))
    table = ax.table(
        cellText=preview.values,
        colLabels=[short_text(column, 16) for column in preview.columns],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    ax.set_title(title, fontsize=10, fontweight="bold")


def plot_auto_dataframe(ax: plt.Axes, frame: pd.DataFrame | None, title: str):
    ax.set_title(title, fontsize=10, fontweight="bold")
    if frame is None or frame.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    data = frame.head(20).copy()
    numbers = numeric_columns(data)
    categories = categorical_columns(data)

    if len(categories) >= 2 and len(numbers) >= 1:
        pivot = data.pivot_table(
            index=categories[0],
            columns=categories[1],
            values=numbers[-1],
            aggfunc="mean",
        )
        if 1 < pivot.shape[0] <= 12 and 1 < pivot.shape[1] <= 8:
            image = ax.imshow(pivot.fillna(0).to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels([short_text(x, 12) for x in pivot.columns], rotation=35, ha="right", fontsize=7)
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels([short_text(x, 18) for x in pivot.index], fontsize=7)
            plt.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
            return

    if len(categories) >= 1 and 2 <= len(numbers) <= 6 and len(data) <= 12:
        plot_data = data.set_index(categories[0])[numbers]
        plot_data.index = plot_data.index.map(lambda value: short_text(value, 16))
        plot_data.plot(kind="bar", ax=ax, width=0.78)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
        return

    if len(categories) >= 1 and len(numbers) >= 1:
        value_col = numbers[-1]
        label_col = categories[0]
        plot_data = data.sort_values(value_col, ascending=True).tail(12)
        labels = plot_data[label_col].map(lambda value: short_text(value, 22))
        ax.barh(labels, plot_data[value_col], color="#2a9d8f")
        ax.set_xlabel(value_col, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="x", alpha=0.25)
        return

    if len(numbers) >= 1:
        values = data[numbers].mean(numeric_only=True).sort_values(ascending=True)
        ax.barh(values.index.map(lambda value: short_text(value, 22)), values.values, color="#457b9d")
        ax.grid(True, axis="x", alpha=0.25)
        return

    plot_table(ax, data, title)


def run_mllib_source(ml_path: Path, output_dir: Path, spark_memory: str) -> Path | None:
    if not ml_path.exists():
        print(f"[ERROR] Cannot find MLlib source: {ml_path}")
        return None

    try:
        configure_spark_environment(spark_memory)
        with pushd(ml_path.parent):
            runpy.run_path(str(ml_path), run_name="__main__")
        plt.close("all")
    except Exception as exc:
        print(f"[ERROR] MLlib source failed: {type(exc).__name__}: {exc}")

    image_path = ml_path.parent / "spotify_regression_result.png"
    if not image_path.exists():
        return None

    copied_path = output_dir / "spotify_mllib_from_source.png"
    copied_path.write_bytes(image_path.read_bytes())
    return copied_path


def render_single_dashboard(
    query_results: list[QueryResult],
    mllib_image: Path | None,
    output_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(18, 18))
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.25, 1.25], hspace=0.42, wspace=0.28)
    fig.suptitle(
        "Spotify Big Data Automated Dashboard - Query + MLlib",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    query_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    for ax, result in zip(query_axes, query_results):
        if result.error:
            ax.axis("off")
            ax.set_title(f"Q{result.spec.index}: {result.spec.title}", fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, short_text(result.error, 110), ha="center", va="center", wrap=True)
        else:
            plot_auto_dataframe(ax, result.frame, f"Q{result.spec.index}: {result.spec.title}")

    for ax in query_axes[len(query_results) :]:
        ax.axis("off")

    ml_ax = fig.add_subplot(grid[2:, :])
    ml_ax.set_title("MLlib Charts From spotify_ML1.py", fontsize=12, fontweight="bold")
    ml_ax.axis("off")
    if mllib_image and mllib_image.exists():
        ml_ax.imshow(mpimg.imread(mllib_image))
    else:
        ml_ax.text(0.5, 0.5, "MLlib chart not available", ha="center", va="center")

    output_path = output_dir / "spotify_bigdata_dashboard.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def dataframe_html(frame: pd.DataFrame | None) -> str:
    if frame is None:
        return "<p>No data.</p>"
    preview = frame.head(20).copy()
    for column in numeric_columns(preview):
        preview[column] = preview[column].map(lambda value: round(float(value), 4) if pd.notna(value) else value)
    return preview.to_html(index=False, escape=True)


def write_html_dashboard(output_dir: Path, dashboard_image: Path, query_results: list[QueryResult]) -> Path:
    table_blocks: list[str] = []
    for result in query_results:
        title = html.escape(f"Q{result.spec.index}: {result.spec.title}")
        body = f"<p class='error'>{html.escape(result.error)}</p>" if result.error else dataframe_html(result.frame)
        table_blocks.append(f"<details><summary>{title}</summary>{body}</details>")

    document = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spotify Big Data Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #1f2933;
      background: #f5f7fa;
    }}
    header {{
      padding: 24px 34px 16px;
      background: #ffffff;
      border-bottom: 1px solid #d8dee8;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    p {{
      color: #667085;
    }}
    main {{
      padding: 22px 34px 40px;
    }}
    section {{
      margin-bottom: 24px;
      padding: 16px;
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #d8dee8;
      border-radius: 6px;
      background: #ffffff;
    }}
    details {{
      margin-top: 10px;
      border: 1px solid #d8dee8;
      border-radius: 6px;
      background: #ffffff;
    }}
    summary {{
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 700;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid #d8dee8;
      padding: 6px 8px;
      text-align: left;
    }}
    th {{
      background: #eef4f7;
    }}
    .error {{
      color: #b42318;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Spotify Big Data Dashboard</h1>
    <p>One automated dashboard generated from HDFS data, 4 SQL queries from spotify_query.py, and MLlib charts from spotify_ML1.py.</p>
  </header>
  <main>
    <section>
      <img src="{html.escape(dashboard_image.name)}" alt="Spotify Big Data dashboard">
    </section>
    <section>
      <h2>Query Data Preview</h2>
      {''.join(table_blocks)}
    </section>
  </main>
</body>
</html>
"""
    html_path = output_dir / "spotify_bigdata_dashboard.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path


def build_dashboard(
    output_dir: Path,
    hdfs_path: str,
    query_limit: int = DEFAULT_QUERY_LIMIT,
    include_query: bool = True,
    include_ml: bool = True,
    spark_memory: str = DEFAULT_SPARK_MEMORY,
    shuffle_partitions: int = DEFAULT_SHUFFLE_PARTITIONS,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_results: list[QueryResult] = []
    mllib_image: Path | None = None

    if include_query:
        print("[STEP] Reading first 4 query definitions from spotify_query.py")
        specs = extract_query_specs(QUERY_SOURCE, query_limit=query_limit)
        spark = build_spark_session("Spotify_Auto_One_Dashboard", spark_memory, shuffle_partitions)
        try:
            print(f"[STEP] Connecting to data source: {hdfs_path}")
            load_tracks_view(spark, hdfs_path)
            query_results = execute_query_specs(spark, specs)
        finally:
            stop_spark_safely(spark)

    if include_ml:
        print("[STEP] Running spotify_ML1.py to generate MLlib charts")
        mllib_image = run_mllib_source(ML_SOURCE, output_dir, spark_memory)

    dashboard_image = render_single_dashboard(query_results, mllib_image, output_dir)
    html_path = write_html_dashboard(output_dir, dashboard_image, query_results)
    print(f"[DONE] Dashboard image: {dashboard_image}")
    print(f"[DONE] Dashboard HTML : {html_path}")
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one automated Spotify Big Data dashboard.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where the single dashboard image and HTML report will be written.",
    )
    parser.add_argument(
        "--hdfs-path",
        default=None,
        help="SpotifyFeatures.csv path. Defaults to the path detected in spotify_query.py.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=DEFAULT_QUERY_LIMIT,
        help="Number of SQL query charts to include. Default: 4.",
    )
    parser.add_argument("--skip-query", action="store_true", help="Do not include SQL query charts.")
    parser.add_argument("--skip-ml", action="store_true", help="Do not include MLlib charts.")
    parser.add_argument(
        "--spark-memory",
        default=DEFAULT_SPARK_MEMORY,
        help="Driver/executor memory for this automation Spark session. Default: 1g.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=DEFAULT_SHUFFLE_PARTITIONS,
        help="Spark SQL shuffle partitions. Lower values reduce memory use. Default: 8.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hdfs_path = args.hdfs_path or extract_hdfs_path(QUERY_SOURCE)
    build_dashboard(
        output_dir=args.output_dir,
        hdfs_path=hdfs_path,
        query_limit=args.query_limit,
        include_query=not args.skip_query,
        include_ml=not args.skip_ml,
        spark_memory=args.spark_memory,
        shuffle_partitions=args.shuffle_partitions,
    )


if __name__ == "__main__":
    main()
