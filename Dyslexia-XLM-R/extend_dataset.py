# Append 20 cross-lingual concept pairs to the dataset.
# Each pair = one new English row + one new Bangla row sharing the same matched_pair_id.

import pandas as pd

df = pd.read_csv("dyslex_probe_stimuli_clean.csv")
df = df.fillna('')
existing_ids = set(df['id'])

# Concept pairs covering common semantic domains
# Format: (en_word, en_ipa, en_syl, bn_word, bn_ipa, bn_syl)
concept_pairs = [
    ("water", "wɔːtər", 2, "পানি", "pani", 2),
    ("fire", "faɪər", 1, "আগুন", "agun", 2),
    ("book", "bʊk", 1, "বই", "bɔi", 1),
    ("tree", "triː", 1, "গাছ", "gatʃʰ", 1),
    ("fish", "fɪʃ", 1, "মাছ", "matʃʰ", 1),
    ("flower", "flaʊər", 2, "ফুল", "pʰul", 1),
    ("sun", "sʌn", 1, "সূর্য", "ʃurdʒo", 2),
    ("moon", "muːn", 1, "চাঁদ", "tʃãd̪", 1),
    ("hand", "hænd", 1, "হাত", "hat̪", 1),
    ("eye", "aɪ", 1, "চোখ", "tʃokʰ", 1),
    ("house", "haʊs", 1, "বাড়ি", "baɽi", 2),
    ("food", "fuːd", 1, "খাবার", "kʰabar", 2),
    ("road", "roʊd", 1, "রাস্তা", "raʃt̪a", 2),
    ("bird", "bɜːrd", 1, "পাখি", "pakʰi", 2),
    ("mother", "mʌðər", 2, "মা", "ma", 1),
    ("father", "fɑːðər", 2, "বাবা", "baba", 2),
    ("child", "tʃaɪld", 1, "শিশু", "ʃiʃu", 2),
    ("school", "skuːl", 1, "স্কুল", "ʃkul", 1),
    ("teacher", "tiːtʃər", 2, "শিক্ষক", "ʃikkʰɔk", 2),
    ("river", "rɪvər", 2, "নদী", "nɔd̪i", 2),
]

new_rows = []
for i, (en_w, en_ipa, en_sc, bn_w, bn_ipa, bn_sc) in enumerate(concept_pairs, start=1):
    match_id = f"XLING_NEW_{i:03d}"
    en_id = f"XL_EN_{i:03d}"
    bn_id = f"XL_BN_{i:03d}"
    
    new_rows.append({
        "id": en_id, "language": "english", "task_category": "concept_match",
        "word": en_w, "ipa": en_ipa, "syllable_count": en_sc,
        "phonological_complexity": "low", "orthographic_transparency": "transparent",
        "voiced_unvoiced": "na", "is_real_word": True,
        "has_consonant_cluster": False, "has_irregular_mapping": False,
        "difficulty": "easy", "pair_id": "", "matched_pair_id": match_id,
        "dissertation_source": "generated", "error_type_targeted": "none",
        "notes": f"cross-lingual concept match: {en_w} ↔ {bn_w}"
    })
    new_rows.append({
        "id": bn_id, "language": "bangla", "task_category": "concept_match",
        "word": bn_w, "ipa": bn_ipa, "syllable_count": bn_sc,
        "phonological_complexity": "low", "orthographic_transparency": "transparent",
        "voiced_unvoiced": "na", "is_real_word": True,
        "has_consonant_cluster": False, "has_irregular_mapping": False,
        "difficulty": "easy", "pair_id": "", "matched_pair_id": match_id,
        "dissertation_source": "generated", "error_type_targeted": "none",
        "notes": f"cross-lingual concept match: {en_w} ↔ {bn_w}"
    })

new_df = pd.DataFrame(new_rows)
df_extended = pd.concat([df, new_df], ignore_index=True)

print(f"Original rows: {len(df)}")
print(f"Added: {len(new_df)} ({len(concept_pairs)} pairs)")
print(f"New total: {len(df_extended)}")
print(f"\nCross-lingual matches now in dataset:")
xling = df_extended[df_extended['matched_pair_id'] != '']
print(f"  Total cross-lingual items: {len(xling)}")
print(f"  Unique pair IDs: {xling['matched_pair_id'].nunique()}")

df_extended.to_csv("dyslex_probe_stimuli_clean.csv", index=False, encoding="utf-8")
print(f"\nSaved updated dataset.")