# Three-Model Comparison

## Model Specs

| Model             | Architecture   | Tokenizer      | Pretrain langs   | Bangla coverage     |   Layers |   Hidden dim | Params   |
|:------------------|:---------------|:---------------|:-----------------|:--------------------|---------:|-------------:|:---------|
| XLM-RoBERTa-large | encoder-only   | SentencePiece  | 100              | ✅ explicit         |       24 |         1024 | 0.56B    |
| Llama 3.1 8B      | decoder-only   | BPE (tiktoken) | mostly English   | ❌ minimal          |       32 |         4096 | 8B       |
| BLOOM 7B1         | decoder-only   | BPE            | 46               | ✅ explicit (ROOTS) |       30 |         4096 | 7.1B     |

## English Probing (RQ1)

| Task     | Metric   | XLM-R Peak   |   XLM-R Score | Llama Peak   |   Llama Score | BLOOM Peak   |   BLOOM Score |
|:---------|:---------|:-------------|--------------:|:-------------|--------------:|:-------------|--------------:|
| voicing  | F1       | L11          |          0.62 | L2           |          0.73 | L6           |          0.55 |
| realword | F1       | L11          |          0.84 | L12          |          0.91 | L22          |          0.79 |
| ortho    | F1       | L0           |          0.92 | L32          |          0.71 | L12          |          0.68 |
| syllable | R²       | L8           |          0.67 | L16          |          0.7  | L29          |          0.68 |

## Cross-lingual Transfer (RQ2)

| Task     | Metric   |   XLM-R BN |   XLM-R Ratio |   Llama BN |   Llama Ratio |   BLOOM BN |   BLOOM Ratio |
|:---------|:---------|-----------:|--------------:|-----------:|--------------:|-----------:|--------------:|
| voicing  | F1       |       0.74 |          1.19 |       0.69 |          0.95 |       0.7  |          0.7  |
| realword | F1       |       0.7  |          0.83 |       0.85 |          0.93 |       0.72 |          0.72 |
| syllable | R²       |       0.52 |          0.78 |       0.18 |          0.26 |       0.57 |          0.57 |

## Cross-lingual RSA (RQ2 confirm)

| Metric             | XLM-R   | Llama   | BLOOM   |
|:-------------------|:--------|:--------|:--------|
| RDM Spearman ρ     |         | 0.41    | 0.21    |
| Top-1 EN→BN acc    |         | 0.85    | 0.65    |
| Peak layer (Top-1) |         | L6      | L4      |

## Ablation Degradation (RQ3)

| Metric                             | XLM-R   | Llama      | BLOOM       |
|:-----------------------------------|:--------|:-----------|:------------|
| voicing  EN drop (top20-baseline)  |         | 0.0        | -0.01       |
| voicing  BN drop                   |         | 0.0        | -0.01       |
| realword EN drop                   |         | -0.02      | -0.004      |
| realword BN drop                   |         | -0.22      | -0.009      |
| syllable EN drop                   |         | -0.16      | -0.005      |
| syllable BN drop                   |         | -0.34      | 0.0         |
| Random-10 control similar to phon? |         | No (clean) | Yes (flat)  |
| Organizational principle           |         | Localized  | Distributed |