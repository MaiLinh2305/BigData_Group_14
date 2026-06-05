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

# Q3: Quantile Thresholds for Pop and Rap (P50, P75, P90)
print("\n [Q3] SUCCESS THRESHOLDS BY PERCENTILE (Pop & Rap)")
spark.sql("""
         WITH filtered_genres AS (SELECT genre, popularity
                                  FROM tracks_deduplicated
                                  WHERE genre IN ('Pop', 'Rap')
                                    AND popularity IS NOT NULL)
         SELECT genre,
                ROUND(PERCENTILE_APPROX(popularity, 0.50), 2) AS P50,
                ROUND(PERCENTILE_APPROX(popularity, 0.75), 2) AS P75,
                ROUND(PERCENTILE_APPROX(popularity, 0.90), 2) AS P90,
                COUNT(*)                                      AS track_count
         FROM filtered_genres
         GROUP BY genre
         ORDER BY genre
         """).show(truncate=False)
# Q4: Audio Feature Analysis: Hit (P90) vs Non-Hit for Pop and Rap
# ============================================================================
print("\n[Q4] KEY AUDIO FEATURES: HIT (P90) vs NON-HIT COMPARISON")
spark.sql("""
         WITH hit_threshold AS (SELECT 'Pop' AS genre, 77 AS hit_threshold
                                UNION ALL
                                SELECT 'Rap', 73),
              feature_comparison AS (SELECT t.genre,
                                            CASE
                                                WHEN t.genre = 'Pop' AND t.popularity >= 77 THEN 'Hit'
                                                WHEN t.genre = 'Rap' AND t.popularity >= 73 THEN 'Hit'
                                                ELSE 'Non-Hit'
                                                END                           AS hit_status,
                                            ROUND(AVG(t.acousticness), 4)     AS acousticness,
                                            ROUND(AVG(t.danceability), 4)     AS danceability,
                                            ROUND(AVG(t.duration_ms), 2)      AS duration_ms,
                                            ROUND(AVG(t.energy), 4)           AS energy,
                                            ROUND(AVG(t.instrumentalness), 6) AS instrumentalness,
                                            ROUND(AVG(t.liveness), 4)         AS liveness,
                                            ROUND(AVG(t.loudness), 2)         AS loudness,
                                            ROUND(AVG(t.speechiness), 6)      AS speechiness,
                                            ROUND(AVG(t.tempo), 2)            AS tempo,
                                            ROUND(AVG(t.valence), 4)          AS valence
                                     FROM tracks_deduplicated t
                                     WHERE t.genre IN ('Pop', 'Rap')
                                       AND t.popularity IS NOT NULL
                                     GROUP BY t.genre, hit_status)
         SELECT genre,
                hit_status,
                acousticness,
                danceability,
                duration_ms,
                energy,
                instrumentalness,
                liveness,
                loudness,
                speechiness,
                tempo,
                valence
         FROM feature_comparison
         ORDER BY genre, hit_status DESC
         """).show(truncate=False)


print("\n [Q4] DIFFERENCE (HIT - NON-HIT) – INDICATOR OF KEY FEATURES (P90 THRESHOLD)")
spark.sql("""
         WITH hit_threshold AS (SELECT 'Pop' AS genre, 77 AS hit_threshold
                                UNION ALL
                                SELECT 'Rap', 73),
              feature_avg AS (SELECT t.genre,
                                     CASE
                                         WHEN t.genre = 'Pop' AND t.popularity >= 77 THEN 'Hit'
                                         WHEN t.genre = 'Rap' AND t.popularity >= 73 THEN 'Hit'
                                         ELSE 'Non-Hit'
                                         END                 AS hit_status,
                                     AVG(t.acousticness)     AS acousticness,
                                     AVG(t.danceability)     AS danceability,
                                     AVG(t.duration_ms)      AS duration_ms,
                                     AVG(t.energy)           AS energy,
                                     AVG(t.instrumentalness) AS instrumentalness,
                                     AVG(t.liveness)         AS liveness,
                                     AVG(t.loudness)         AS loudness,
                                     AVG(t.speechiness)      AS speechiness,
                                     AVG(t.tempo)            AS tempo,
                                     AVG(t.valence)          AS valence
                              FROM tracks_deduplicated t
                              WHERE t.genre IN ('Pop', 'Rap')
                                AND t.popularity IS NOT NULL
                              GROUP BY t.genre, hit_status)
         SELECT genre,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN acousticness END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN acousticness END), 4)     AS diff_acousticness,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN danceability END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN danceability END), 4)     AS diff_danceability,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN duration_ms END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN duration_ms END), 2)      AS diff_duration_ms,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN energy END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN energy END), 4)           AS diff_energy,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN instrumentalness END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN instrumentalness END), 6) AS diff_instrumentalness,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN liveness END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN liveness END), 4)         AS diff_liveness,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN loudness END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN loudness END), 2)         AS diff_loudness,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN speechiness END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN speechiness END), 6)      AS diff_speechiness,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN tempo END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN tempo END), 2)            AS diff_tempo,
                ROUND(AVG(CASE WHEN hit_status = 'Hit' THEN valence END) -
                      AVG(CASE WHEN hit_status = 'Non-Hit' THEN valence END), 4)          AS diff_valence
         FROM feature_avg
         GROUP BY genre
         ORDER BY genre
         """).show(truncate=False)


spark.stop()











