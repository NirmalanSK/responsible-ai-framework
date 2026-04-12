# 📈 Simulation Baseline Report

**Generated:** April 2026  
**Framework:** Responsible AI Framework v5.0 (pipeline_v15g)  
**Total Labeled Queries:** 16  
**Overall Accuracy:** 16/16 (100.0%)  

## Domain-Wise Precision / Recall / F1

| Domain | N | TP | FP | TN | FN | Precision | Recall | F1 | FPR |
|--------|---|----|----|----|----|-----------|--------|----|-----|
| audit_gap | 2 | 2 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| criminal_justice_bias | 1 | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| financial_fraud | 4 | 4 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| harassment | 1 | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| hate_speech | 1 | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| healthcare_bias | 1 | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| privacy_violation | 3 | 3 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| representation_bias | 3 | 3 | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 | 0.0% |
| **OVERALL** | **16** | **16** | **0** | **0** | **0** | **100.0%** | **100.0%** | **1.00** | **0.0%** |

## Year 2 Targets

| Metric | Year 1 (16-case AIAAIC) | Year 2 Target | Method |
|--------|-----------------|---------------|--------|
| Overall F1 | 1.00 (small labeled set) | 0.92+ (full 2,223-case) | Bayesian Optimization on AIAAIC 2,223 |
| HarmBench Recall | 14.5% | 75-80% | XLM-RoBERTa semantic |
| FPR | 0.0% | <2% | SBERT hybrid tiering |
| DAG input | Manual | Auto | DoWhy integration |
| Labeled test coverage | 16/50 (32%) | 2,223/2,223 (100%) | Full AIAAIC ground-truth labeling |
