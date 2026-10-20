# Committed evaluation and demo data

These are test-only exports of the exact original evaluation partitions, not new training data.

- `cmapss_endpoints.csv`: 100 official final test endpoints, causal engineered features, engine/cycle and raw official RUL.
- `cmapss_replay.csv`: complete observed test prefixes for FD001 engines 1, 20 and 81. Columns: unit, cycle, setting_1..3, s1..s21.
- `ai4i_holdout.csv`: 2,000 original AI4I holdout rows, in the saved model's test-index order.
- `ai4i_presets.json`: a small selection of those held-out snapshots, with physical telemetry only.

The evaluator compares causal preprocessing against the frozen endpoint features on every replay engine. Full 100-engine raw-feature regeneration remains available with the optional original NASA download. No feature export uses test labels as inputs.

Sources: NASA C-MAPSS, Saxena et al. (2008), https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data ; UCI AI4I 2020, DOI 10.24432/C5HS5C, CC BY 4.0, https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset . The AI4I CSV is an attributed subset exported without changing original values. Source download hashes are in reports/data_provenance.json. Dataset terms remain separate from the code license.
