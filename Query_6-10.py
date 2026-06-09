# Q6: Undiscovered high-potential artists (top 5 Pop and top 5 Rap)
print("\n [Q6] UNDISCOVERED HIGH-POTENTIAL ARTISTS (TOP 5 POP & TOP 5 RAP)")
spark.sql("""
          WITH artist_stats AS (SELECT TRIM(artist_name)         AS artist_name,
                                       genre,
                                       COUNT(*)                  AS num_tracks,
                                       ROUND(AVG(popularity), 2) AS avg_popularity,
                                       MAX(popularity)           AS max_popularity,
                                       ROW_NUMBER()                 OVER (PARTITION BY genre ORDER BY AVG(popularity) DESC, COUNT(*) ASC) AS rank_in_genre
                                FROM tracks_deduplicated
                                WHERE genre IN ('Pop', 'Rap')
                                  AND artist_name IS NOT NULL
                                  AND TRIM(artist_name) != ''
              AND popularity IS NOT NULL
          GROUP BY TRIM (artist_name), genre
          HAVING COUNT (*) BETWEEN 1
             AND 5
             AND AVG (popularity) >= 60
              )
               , genre_p90 AS (
          SELECT genre, PERCENTILE_APPROX(popularity, 0.90) AS p90_threshold
          FROM tracks_deduplicated
          WHERE genre IN ('Pop', 'Rap')
          GROUP BY genre
              )
          SELECT a.artist_name,
                 a.genre,
                 a.num_tracks,
                 a.avg_popularity,
                 a.max_popularity,
                 ROUND(g.p90_threshold, 2)                                                          AS genre_p90,
                 CASE WHEN a.max_popularity >= g.p90_threshold THEN 'Has Hit' ELSE 'No Hit Yet' END AS hit_status
          FROM artist_stats a
                   JOIN genre_p90 g ON a.genre = g.genre
          WHERE a.max_popularity < g.p90_threshold
            AND a.rank_in_genre <= 5
          ORDER BY a.genre, a.rank_in_genre
          """).show(truncate=False)

# Q7: Edge effect – comparing almost-hit (P75 to P90) vs real hit (>=P90)
print("\n[Q9] EDGE EFFECT: ALMOST-HIT vs REAL HIT (POP & RAP)")
spark.sql("""
          WITH hit_thresholds AS (SELECT 'Pop' AS genre, 72 AS p75, 77 AS p90
                                  UNION ALL
                                  SELECT 'Rap', 63, 73),
               feature_comparison AS (SELECT t.genre,
                                             CASE
                                                 WHEN (t.genre = 'Pop' AND t.popularity >= 77) OR
                                                      (t.genre = 'Rap' AND t.popularity >= 73) THEN 'Real Hit'
                                                 WHEN (t.genre = 'Pop' AND t.popularity BETWEEN 72 AND 76) OR
                                                      (t.genre = 'Rap' AND t.popularity BETWEEN 63 AND 72)
                                                     THEN 'Almost Hit'
                                                 ELSE NULL
                                                 END                       AS hit_status,
                                             ROUND(AVG(t.duration_ms), 2)  AS duration_ms,
                                             ROUND(AVG(t.danceability), 4) AS danceability,
                                             ROUND(AVG(t.energy), 4)       AS energy,
                                             ROUND(AVG(t.loudness), 2)     AS loudness,
                                             ROUND(AVG(t.tempo), 2)        AS tempo,
                                             ROUND(AVG(t.valence), 4)      AS valence,
                                             ROUND(AVG(t.speechiness), 6)  AS speechiness,
                                             COUNT(*)                      AS track_count
                                      FROM tracks_deduplicated t
                                               JOIN hit_thresholds h ON t.genre = h.genre
                                      WHERE t.genre IN ('Pop', 'Rap')
                                        AND t.popularity IS NOT NULL
                                        AND t.popularity >= h.p75 -- chỉ lấy từ P75 trở lên
                                      GROUP BY t.genre, hit_status)
          SELECT genre,
                 hit_status,
                 track_count,
                 duration_ms,
                 danceability,
                 energy,
                 loudness,
                 tempo,
                 valence,
                 speechiness
          FROM feature_comparison
          WHERE hit_status IS NOT NULL
          ORDER BY genre, hit_status
          """).show(truncate=False)

print("\n[Q7] DIFFERENCE (REAL HIT - ALMOST HIT) – DECISIVE FACTORS")
spark.sql("""
          WITH hit_thresholds AS (SELECT 'Pop' AS genre, 72 AS p75, 77 AS p90
                                  UNION ALL
                                  SELECT 'Rap', 63, 73),
               feature_avg AS (SELECT t.genre,
                                      CASE
                                          WHEN (t.genre = 'Pop' AND t.popularity >= 77) OR
                                               (t.genre = 'Rap' AND t.popularity >= 73) THEN 'Real Hit'
                                          WHEN (t.genre = 'Pop' AND t.popularity BETWEEN 72 AND 76) OR
                                               (t.genre = 'Rap' AND t.popularity BETWEEN 63 AND 72) THEN 'Almost Hit'
                                          END             AS hit_status,
                                      AVG(t.duration_ms)  AS duration_ms,
                                      AVG(t.danceability) AS danceability,
                                      AVG(t.energy)       AS energy,
                                      AVG(t.loudness)     AS loudness,
                                      AVG(t.tempo)        AS tempo,
                                      AVG(t.valence)      AS valence,
                                      AVG(t.speechiness)  AS speechiness
                               FROM tracks_deduplicated t
                                        JOIN hit_thresholds h ON t.genre = h.genre
                               WHERE t.genre IN ('Pop', 'Rap')
                                 AND t.popularity >= h.p75
                               GROUP BY t.genre, hit_status)
          SELECT genre,
                 ROUND(MAX(CASE WHEN hit_status = 'Real Hit' THEN duration_ms END) -
                       MAX(CASE WHEN hit_status = 'Almost Hit' THEN duration_ms END), 2)  AS diff_duration_ms,
                 ROUND(MAX(CASE WHEN hit_status = 'Real Hit' THEN danceability END) -
                       MAX(CASE WHEN hit_status = 'Almost Hit' THEN danceability END), 4) AS diff_danceability,
                 ROUND(MAX(CASE WHEN hit_status = 'Real Hit' THEN energy END) -
                       MAX(CASE WHEN hit_status = 'Almost Hit' THEN energy END), 4)       AS diff_energy,
                 ROUND(MAX(CASE WHEN hit_status = 'Real Hit' THEN loudness END) -
                       MAX(CASE WHEN hit_status = 'Almost Hit' THEN loudness END), 2)     AS diff_loudness,
                 ROUND(MAX(CASE WHEN hit_status = 'Real Hit' THEN tempo END) -
                       MAX(CASE WHEN hit_status = 'Almost Hit' THEN tempo END), 2)        AS diff_tempo,
                 ROUND(MAX(CASE WHEN hit_status = 'Real Hit' THEN valence END) -
                       MAX(CASE WHEN hit_status = 'Almost Hit' THEN valence END), 4)      AS diff_valence,
                 ROUND(MAX(CASE WHEN hit_status = 'Real Hit' THEN speechiness END) -
                       MAX(CASE WHEN hit_status = 'Almost Hit' THEN speechiness END), 6)  AS diff_speechiness
          FROM feature_avg
          WHERE hit_status IS NOT NULL
          GROUP BY genre
          ORDER BY genre
          """).show(truncate=False)

# Tính phân vị P50, P75, P90 cho thể loại R&B
print("\n [Q8-PREP] PERCENTILES FOR R&B")
spark.sql("""
          WITH rnb_tracks AS (SELECT popularity
                              FROM tracks_deduplicated
                              WHERE genre = 'R&B'
                                AND popularity IS NOT NULL)
          SELECT 'R&B'                                         AS genre,
                 ROUND(PERCENTILE_APPROX(popularity, 0.50), 2) AS P50,
                 ROUND(PERCENTILE_APPROX(popularity, 0.75), 2) AS P75,
                 ROUND(PERCENTILE_APPROX(popularity, 0.90), 2) AS P90,
                 COUNT(*)                                      AS track_count
          FROM rnb_tracks
          """).show(truncate=False)

# Q8 for R&B: popularity < 49 (low popularity)
print("\n [Q8] R&B SLEEPER TRACKS (POPULARITY < 49, CLOSE TO HIT PROFILE)")
spark.sql("""
          WITH hit_avg AS (SELECT AVG(danceability) AS avg_dance,
                                  AVG(energy)       AS avg_energy,
                                  AVG(loudness)     AS avg_loud,
                                  AVG(tempo)        AS avg_tempo,
                                  AVG(duration_ms)  AS avg_dur
                           FROM tracks_deduplicated
                           WHERE genre = 'R&B'
                             AND popularity >= 54),
               track_distance AS (SELECT t.track_name,
                                         t.artist_name,
                                         t.popularity,
                                         SQRT(
                                                 POWER((t.danceability - h.avg_dance) / NULLIF(h.avg_dance, 0), 2) +
                                                 POWER((t.energy - h.avg_energy) / NULLIF(h.avg_energy, 0), 2) +
                                                 POWER((t.loudness - h.avg_loud) / NULLIF(ABS(h.avg_loud), 0), 2) +
                                                 POWER((t.tempo - h.avg_tempo) / NULLIF(h.avg_tempo, 0), 2) +
                                                 POWER((t.duration_ms - h.avg_dur) / NULLIF(h.avg_dur, 0), 2)
                                         ) AS distance_to_hit
                                  FROM tracks_deduplicated t
                                           CROSS JOIN hit_avg h
                                  WHERE t.genre = 'R&B'
                                    AND t.popularity IS NOT NULL
                                    AND t.popularity < 49)
          SELECT track_name, artist_name, popularity, ROUND(distance_to_hit, 4) AS audio_distance
          FROM track_distance
          WHERE distance_to_hit IS NOT NULL
          ORDER BY distance_to_hit ASC LIMIT 5
          """).show(truncate=False)

print("\n [Q9] HIT RATE BY MUSICAL KEY (POP & RAP) - TOP 5")
for genre in ['Pop', 'Rap']:
    threshold = 77 if genre == 'Pop' else 73
    print(f"\n--- {genre} (Hit threshold = {threshold}) ---")
    spark.sql(f"""
  WITH key_stats AS (
      SELECT
          key,
          COUNT(*) AS total_tracks,
          ROUND(SUM(CASE WHEN popularity >= {threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS hit_rate
      FROM tracks_deduplicated
      WHERE genre = '{genre}' AND key IS NOT NULL AND key != ''
      GROUP BY key
      HAVING COUNT(*) >= 30
  )
  SELECT key, total_tracks, hit_rate
  FROM key_stats
  ORDER BY hit_rate DESC
  LIMIT 5

  """).show(truncate=False)

print("\n [Q10] HIT RATE BY TIME SIGNATURE (POP & RAP)")
for genre in ['Pop', 'Rap']:
    threshold = 77 if genre == 'Pop' else 73
    print(f"\n--- {genre} (Hit threshold = {threshold}) ---")
    spark.sql(f"""
  WITH ts_stats AS (
      SELECT
          time_signature,
          COUNT(*) AS total_tracks,
          ROUND(SUM(CASE WHEN popularity >= {threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS hit_rate
      FROM tracks_deduplicated
      WHERE genre = '{genre}' AND time_signature IS NOT NULL AND time_signature != ''
      GROUP BY time_signature
      HAVING COUNT(*) >= 30
  )
  SELECT time_signature, total_tracks, hit_rate
  FROM ts_stats
  ORDER BY hit_rate DESC

  """).show(truncate=False)

spark.stop()
