from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.window import Window

# Initialize Spark Session for Batch Processing
spark = SparkSession.builder \
    .appName("Spotify_HDFS_Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define Data Schema
schema = StructType([
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
    StructField("valence", DoubleType(), True)
])

# Load Data from HDFS
print("Loading data from HDFS...")
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .schema(schema) \
    .load("hdfs://localhost:9000/spark_data/SpotifyFeatures.csv")

# Count raw records
raw_count = df.count()
print(f"Total raw records loaded: {raw_count}")

df.printSchema()


# Data Preprocessing: Deduplication by track_id
window_spec = Window.partitionBy("track_id").orderBy(col("genre").asc())
df_dedup = df.withColumn("rn", row_number().over(window_spec)) \
    .filter(col("rn") == 1) \
    .drop("rn")

# Count unique records
unique_count = df_dedup.count()
print(f"Total unique tracks identified: {unique_count}")

df_dedup.createOrReplaceTempView("tracks_deduplicated")

print("\n" + "=" * 80)
print("SPOTIFY MARKET INTELLIGENCE BATCH REPORT")
print(f"DATA SOURCE: HDFS | TOTAL TRACKS: {unique_count}")
print("=" * 80)

# Q1: Market Opportunity
print("\n[Q1] MARKET OPPORTUNITY ANALYSIS")
spark.sql("""
          WITH genre_metrics AS (SELECT genre,
                                        COUNT(*)                     AS track_count,
                                        ROUND(AVG(popularity), 2)    AS avg_popularity,
                                        ROUND(STDDEV(popularity), 2) AS popularity_stddev
                                 FROM tracks_deduplicated
                                 WHERE genre IS NOT NULL
                                   AND TRIM(genre) != '' AND popularity IS NOT NULL
          GROUP BY genre
          HAVING COUNT (*) >= 100
             AND AVG (popularity)
               > 30
              )
               , market_total AS (
          SELECT SUM (track_count) AS total_tracks
          FROM genre_metrics)
          SELECT genre,
                 track_count,
                 avg_popularity,
                 ROUND(avg_popularity * (1 - (track_count * 1.0 / NULLIF(total_tracks, 0))), 2) AS opportunity_score
          FROM genre_metrics,
               market_total
          WHERE track_count < (SELECT AVG(track_count) FROM genre_metrics)
          ORDER BY opportunity_score DESC
          """).show(10, truncate=False)
# Q2: Market Benchmark
print("\n[Q2] TOP PERFORMING TRACKS IN TOP GENRES")
spark.sql("""
         WITH genre_avg AS (SELECT genre, AVG(popularity) AS genre_avg_pop
                            FROM tracks_deduplicated
                            WHERE popularity IS NOT NULL
                              AND genre IS NOT NULL
                              AND genre IN ('Pop', 'Rock', 'Rap')
                            GROUP BY genre
                            HAVING COUNT(*) >= 500
                               AND AVG(popularity) > 35),
              artist_stats AS (SELECT genre,
                                      TRIM(track_name)  AS track_name,
                                      TRIM(artist_name) AS artist_name,
                                      AVG(popularity)   AS track_avg_pop,
                                      COUNT(*)          AS total_tracks,
                                      ROW_NUMBER()         OVER (
             PARTITION BY genre
             ORDER BY AVG(popularity) DESC
         ) AS rank_in_genre
                               FROM tracks_deduplicated
                               WHERE popularity IS NOT NULL
                                 AND artist_name IS NOT NULL
                                 AND TRIM(artist_name) != ''
             AND genre IS NOT NULL
             AND genre IN ('Pop', 'Rock', 'Rap')
         GROUP BY genre, TRIM (track_name), TRIM (artist_name)
         HAVING COUNT (*) >= 1
             )
         SELECT a.genre,
                a.track_name,
                a.artist_name,
                ROUND(a.track_avg_pop, 2)                   AS track_popularity,
                ROUND(g.genre_avg_pop, 2)                   AS genre_avg,
                ROUND(a.track_avg_pop - g.genre_avg_pop, 2) AS delta_vs_genre
         FROM artist_stats a
                  JOIN genre_avg g
                       ON a.genre = g.genre
         WHERE a.rank_in_genre <= 3
         ORDER BY a.genre, delta_vs_genre DESC
         """).show(9, truncate=False)

spark.stop()











