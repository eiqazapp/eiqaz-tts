# niletts-4h-tashkeel-ai-v1

Egyptian-Arabic tashkeel'd transcripts for the NileTTS 4h corpus, produced by
a rule-validated LLM pipeline (Track C), intended to replace catt_eo output
as the text source for MixerTTS from-scratch training on Kaggle T4.

- Scope: NileTTS 4h train pool units (21,880 total input; this package holds
  the units whose tashkeel passed all engine gates).
- Engine: ph3_engine.mjs v5.1 (guide v2.1, approved 2026-09-30). Gates:
  exact letter/punctuation skeleton, closed-list tanween, taa-marbuta no
  haraka, density floor, deterministic Tier A + rabbena enforcement,
  letter-restore, quarantine skip.
- Punctuation: Arabic ، ؟ converted to ASCII , ? BEFORE tashkeel (the train
  kernel's symbol table only supports . , ? ! as ids 5-8).
- NO audio in this package. No manifest replacement — text only.
- Excluded by design: quarantined units (user-approved 2026-09-30), units
  still failing gates (see review_needed.json), units without records.

## Current build (partial runs are marked)

- input units: 21880
- included (ok): 65  (train 60 / eval 5)
- failed (under review): 45
- quarantined: 6
- no record yet: 21764
- diacritic density median: 3.86 marks/word

Regeneration: scripts/ + work/ artifacts are archived alongside; the engine
is deterministic post-LLM (maps + fixes logged per unit in the JSONL).
