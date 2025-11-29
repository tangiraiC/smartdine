# SmartDine Dataset Log

## Version: v1
**Generated:** 2025-10-30 22:00 EST  
**Seed:** 42  
**Source file:** recommender/data/raw/filter_all_t.json  

| Split | Rows | Columns | File Path | Size (MB) |
|--------|-------|----------|------------|-----------|
| train  | 87,013 | 6 | recommender/data/processed/train.parquet | 152 |
| val    | 10,860 | 6 | recommender/data/processed/val.parquet | 19 |
| test   | 11,015 | 6 | recommender/data/processed/test.parquet | 20 |

**Image Manifest**
- Unique pic_id count: ~200,000  
- Manifest path: recommender/data/processed/image_manifest.parquet  

**Validation:** All checks passed (see validation_checklist.md)  
**Notes:**  
- 3% of rows had empty `review_text`.  
- 0.2% missing pics; left as empty lists.  
- All ratings ∈ [1–5]; verified.  
- Next version (v2) may normalize price-level metadata.

---
