# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Import libraries and initialize Spark Session
# ─────────────────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
   StringIndexer, OneHotEncoder,
   VectorAssembler, StandardScaler
)
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os


spark = SparkSession.builder \
   .appName("Spotify_Popularity_Regression") \
   .getOrCreate()


spark.sparkContext.setLogLevel("WARN")


HDFS_PATH = "hdfs://localhost:9000/spark_data/"
print("Spark Session initialized successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Read Spotify data (from local filesystem)
# ─────────────────────────────────────────────────────────────────────────────
df_spotify = spark.read.csv(
   HDFS_PATH + "SpotifyFeatures.csv",
   header=True,
   inferSchema=True
)


print("\n── Initial Schema ──────────────────────────────────────")
df_spotify.printSchema()
print(f"Total records: {df_spotify.count():,}")
df_spotify.show(5)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: Filter Pop & Rap, cast types, handle missing values
# ─────────────────────────────────────────────────────────────────────────────


# Filter target genres
df_target = df_spotify.filter(col("genre").isin(["Pop", "Rap"]))
print(f"\n── Genre Distribution ─────────────────────────────────")
print(f"Total Pop + Rap: {df_target.count():,} songs")
df_target.groupBy("genre").count().orderBy("genre").show()


# Cast numeric columns
numeric_cols = [
   "danceability", "energy", "loudness", "tempo",
   "speechiness", "valence", "duration_ms",
   "acousticness", "liveness", "instrumentalness"
]
for c in numeric_cols:
   df_target = df_target.withColumn(c, col(c).cast("double"))


# Handle missing values
df_target = df_target.dropna(subset=numeric_cols + ["key", "mode", "time_signature"])


# Feature engineering: convert duration_ms → duration_s
df_target = df_target.withColumn("duration_s", col("duration_ms") / 1000.0)


print(f"After cleaning: {df_target.count():,} valid songs")


print("\n── Popularity Statistics (Pop + Rap) ──────────────────")
df_target.select("popularity").describe().show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Build distributed preprocessing Pipeline (Spark MLlib)
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: StringIndexer for 3 categorical columns
key_indexer   = StringIndexer(inputCol="key",            outputCol="key_idx",   handleInvalid="keep")
mode_indexer  = StringIndexer(inputCol="mode",           outputCol="mode_idx",  handleInvalid="keep")
ts_indexer    = StringIndexer(inputCol="time_signature", outputCol="ts_idx",    handleInvalid="keep")


# Step 2: OneHotEncoder (dropLast=True, handleInvalid="keep" for safety)
ohe = OneHotEncoder(
   inputCols=["key_idx", "mode_idx", "ts_idx"],
   outputCols=["key_ohe", "mode_ohe", "ts_ohe"],
   dropLast=True,
   handleInvalid="keep"      # ADD THIS LINE TO AVOID ERRORS WHEN TEST HAS NEW VALUES
)


# Step 3: VectorAssembler
numeric_features = [
   "danceability", "energy", "loudness", "tempo",
   "speechiness", "valence", "duration_s",
   "acousticness", "liveness", "instrumentalness"
]
categorical_ohe = ["key_ohe", "mode_ohe", "ts_ohe"]
all_feature_cols = numeric_features + categorical_ohe


assembler = VectorAssembler(
   inputCols=all_feature_cols,
   outputCol="raw_features",
   handleInvalid="skip"
)


# Step 4: StandardScaler
scaler = StandardScaler(
   inputCol="raw_features", outputCol="features",
   withStd=True, withMean=True
)


print("Pipeline stages defined:")
print(f"  • StringIndexer : key, mode, time_signature")
print(f"  • OneHotEncoder : dropLast=True, handleInvalid='keep'")
print(f"  • VectorAssembler: {len(numeric_features)} numeric + 3 OHE vectors")
print(f"  • StandardScaler : withMean=True, withStd=True")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Train / Test split and model training via Pipeline
# ─────────────────────────────────────────────────────────────────────────────
train_df, test_df = df_target.randomSplit([0.8, 0.2], seed=42)
print(f"\n── Dataset Split ──────────────────────────────────────")
print(f"  Train : {train_df.count():,} songs  ({train_df.count()/df_target.count()*100:.1f}%)")
print(f"  Test  : {test_df.count():,} songs  ({test_df.count()/df_target.count()*100:.1f}%)")


# Model 1: Linear Regression (Elastic Net)
lr = LinearRegression(
   featuresCol="features", labelCol="popularity",
   maxIter=50, regParam=0.1, elasticNetParam=0.5
)
pipeline_lr = Pipeline(stages=[
   key_indexer, mode_indexer, ts_indexer,
   ohe, assembler, scaler, lr
])
print("\nTraining Linear Regression Pipeline...")
lr_pipeline_model = pipeline_lr.fit(train_df)
print("Linear Regression training complete.")


# Model 2: GBT Regressor
gbt = GBTRegressor(
   featuresCol="features", labelCol="popularity",
   maxIter=50, maxDepth=5, stepSize=0.1, seed=42
)
pipeline_gbt = Pipeline(stages=[
   key_indexer, mode_indexer, ts_indexer,
   ohe, assembler, scaler, gbt
])
print("Training GBT Regressor Pipeline...")
gbt_pipeline_model = pipeline_gbt.fit(train_df)
print("GBT Regressor training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: Model evaluation – RMSE, MAE, R²
# ─────────────────────────────────────────────────────────────────────────────
evaluator_rmse = RegressionEvaluator(labelCol="popularity", predictionCol="prediction", metricName="rmse")
evaluator_mae  = RegressionEvaluator(labelCol="popularity", predictionCol="prediction", metricName="mae")
evaluator_r2   = RegressionEvaluator(labelCol="popularity", predictionCol="prediction", metricName="r2")


pred_lr  = lr_pipeline_model.transform(test_df)
pred_gbt = gbt_pipeline_model.transform(test_df)


rmse_lr = evaluator_rmse.evaluate(pred_lr);  mae_lr = evaluator_mae.evaluate(pred_lr);  r2_lr = evaluator_r2.evaluate(pred_lr)
rmse_gbt= evaluator_rmse.evaluate(pred_gbt); mae_gbt= evaluator_mae.evaluate(pred_gbt); r2_gbt= evaluator_r2.evaluate(pred_gbt)


print("\n══════════════════════════════════════════════════════")
print("  MODEL EVALUATION RESULTS ON TEST SET")
print("══════════════════════════════════════════════════════")
print(f"{'Model':<28} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print(f"{'─'*28} {'─'*8} {'─'*8} {'─'*8}")
print(f"{'Linear Regression (Elastic Net)':<28} {rmse_lr:>8.4f} {mae_lr:>8.4f} {r2_lr:>8.4f}")
print(f"{'GBT Regressor':<28} {rmse_gbt:>8.4f} {mae_gbt:>8.4f} {r2_gbt:>8.4f}")
print("══════════════════════════════════════════════════════")


print("\n── Sample GBT Predictions (first 10 songs) ───────────")
pred_gbt.select("genre", "track_name", "popularity", "prediction") \
       .withColumn("prediction", col("prediction").cast("double")) \
       .show(10, truncate=30)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: Result visualization
# ─────────────────────────────────────────────────────────────────────────────
pd_lr  = pred_lr.select("popularity", "prediction").toPandas()
pd_gbt = pred_gbt.select("popularity", "prediction").toPandas()


# --- Get Feature Importance from GBT and build accurate feature names ---
gbt_model_stage = gbt_pipeline_model.stages[-1]
importances = gbt_model_stage.featureImportances.toArray()
n_features = len(importances)


# Get fitted models from pipeline
key_idx_model  = gbt_pipeline_model.stages[0]
mode_idx_model = gbt_pipeline_model.stages[1]
ts_idx_model   = gbt_pipeline_model.stages[2]


# OHE dimensions after dropLast=True = n_labels - 1
ohe_key_dim  = len(key_idx_model.labels)  - 1
ohe_mode_dim = len(mode_idx_model.labels) - 1
ohe_ts_dim   = len(ts_idx_model.labels)   - 1


# Build feature name list
feature_names = numeric_features.copy()  # 10 numeric features


feature_names += [f"key_{key_idx_model.labels[i]}"   for i in range(ohe_key_dim)]
feature_names += [f"mode_{mode_idx_model.labels[i]}" for i in range(ohe_mode_dim)]
feature_names += [f"ts_{ts_idx_model.labels[i]}"     for i in range(ohe_ts_dim)]


# Fallback if lengths don't match
if len(feature_names) != n_features:
   n_ohe = n_features - len(numeric_features)
   feature_names = numeric_features + [f"ohe_{i}" for i in range(n_ohe)]


feat_imp_df = pd.DataFrame({
   "feature":    feature_names[:n_features],
   "importance": importances
}).sort_values("importance", ascending=False)


feat_imp_top = feat_imp_df.head(15).sort_values("importance", ascending=True)


# ── Plot charts ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Popularity Prediction Model Analysis – Pop & Rap Spotify",
            fontsize=15, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)


ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])


# Chart 1: Scatter LR
ax1.scatter(pd_lr['popularity'], pd_lr['prediction'], alpha=0.15, s=6, color='steelblue')
lim = [pd_lr['popularity'].min(), pd_lr['popularity'].max()]
ax1.plot(lim, lim, 'r--', lw=1.5, label='Ideal')
ax1.set_xlabel('Actual Popularity'); ax1.set_ylabel('Predicted Popularity')
ax1.set_title(f'Linear Regression\nRMSE={rmse_lr:.3f} | R²={r2_lr:.4f}')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)


# Chart 2: Scatter GBT
ax2.scatter(pd_gbt['popularity'], pd_gbt['prediction'], alpha=0.15, s=6, color='seagreen')
ax2.plot(lim, lim, 'r--', lw=1.5, label='Ideal')
ax2.set_xlabel('Actual Popularity'); ax2.set_ylabel('Predicted Popularity')
ax2.set_title(f'GBT Regressor\nRMSE={rmse_gbt:.3f} | R²={r2_gbt:.4f}')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)


# Chart 3: Compare RMSE / MAE / R²
metrics  = ['RMSE', 'MAE', 'R²']
vals_lr  = [rmse_lr,  mae_lr,  r2_lr]
vals_gbt = [rmse_gbt, mae_gbt, r2_gbt]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax3.bar(x - w/2, vals_lr,  w, label='Linear Regression', color='steelblue', alpha=0.8)
bars2 = ax3.bar(x + w/2, vals_gbt, w, label='GBT Regressor',     color='seagreen',  alpha=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(metrics)
ax3.set_title('Evaluation Metrics Comparison'); ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars1:
   ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
   ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)


# Chart 4: Feature Importance (top 15)
colors = ['#e74c3c' if i >= len(feat_imp_top) - 3 else '#3498db'
         for i in range(len(feat_imp_top))]
bars = ax4.barh(feat_imp_top['feature'], feat_imp_top['importance'], color=colors)
ax4.set_xlabel('Importance score'); ax4.set_title('Top 15 Feature Importance (GBT)')
ax4.grid(True, alpha=0.3, axis='x')
for bar in bars:
   ax4.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():.4f}', va='center', fontsize=7)


# Remove auto-title added by matplotlib's boxplot
fig.texts = [t for t in fig.texts if 'Boxplot' not in t.get_text()]


plt.savefig('spotify_regression_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Chart saved: spotify_regression_result.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8: Stop Spark Session
# ─────────────────────────────────────────────────────────────────────────────
spark.stop()
print("Spark Session stopped.")




