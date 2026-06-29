# ARIA Benchmark Diagnosis Request

## Purpose

This file is for the coding assistant to review the current ARIA empirical evaluation results and diagnose whether the benchmark is valid or whether the reported scores are inflated due to leakage, overly easy evaluation logic, small sample size, or incorrect prediction/ground-truth construction.

The current goal is **not** to improve the model immediately.  
The first goal is to verify that the evaluation pipeline is honest and reliable.

---

## Current Reported Results

### Vision Modality Evaluation — FER2013 Photos

| Model / Architecture | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| ARIA Vision Emotion Model | 0.9948 | 0.9946 | 0.9947 | 0.9946 |

### Audio/Prosody Modality Evaluation — RAVDESS `.wav`

| Model / Architecture | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| ARIA Audio Prosody Model | 0.9861 | 0.9922 | 0.9887 | 0.9917 |

### Text/Semantic Modality Evaluation — Mohler Parquet

| Model / Architecture | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| ARIA Semantic Grading Model | 0.7331 | 0.5230 | 0.5709 | 0.7840 |

### Multimodal Fusion Engine Comparison

| Model / Architecture | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| ARIA Multimodal Attention Fusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ARIA Unweighted Concatenation Baseline | 0.8333 | 0.7500 | 0.7333 | 0.7500 |

---

## Diagnosis Summary

The results are suspiciously high for Vision, Audio/Prosody, and Multimodal Fusion.

The most suspicious values are:

```text
Vision FER2013 accuracy:        0.9946
Audio RAVDESS accuracy:         0.9917
Attention Fusion accuracy:      1.0000
```

These are not impossible, but they are unlikely unless the benchmark is very small, simplified, or leaking labels into predictions.

The semantic result looks more realistic:

```text
Semantic Mohler accuracy:       0.7840
Semantic Mohler F1-score:       0.5709
Semantic Mohler recall:         0.5230
```

The semantic result should be treated as the most believable until the other benchmarks are validated.

---

## Primary Concern

The evaluation may be measuring whether the code can reconstruct labels from metadata, filenames, folder names, or precomputed fields instead of measuring true model inference.

The coding assistant must inspect the benchmark code and confirm:

```text
y_true comes only from ground-truth annotations
y_pred comes only from model inference outputs
y_pred is not copied, remapped, or derived from y_true
```

---

# Required Files To Inspect

Please inspect these files first:

```text
tests/benchmarks/run_baseline_benchmarks.py
modules/module_04_fusion/attention_fusion.py
modules/module_04_fusion/fusion_engine.py
modules/module_04_fusion/concat_fusion.py
modules/module_04_fusion/schema.py
modules/module_04_fusion/normalizer.py
modules/module_02_vision/emotion.py
modules/module_02_vision/vision_processor.py
modules/module_3_prosody/extractor.py
modules/module_3_prosody/pipeline.py
```

If the repo uses `module_03_prosody` instead of `module_3_prosody`, inspect the actual existing folder.

---

# Specific Leakage Checks

## 1. Vision / FER2013 Leakage Checks

Inspect the FER2013 evaluation code.

Check whether the prediction is accidentally created from:

```text
folder name
file path
CSV label
ground-truth emotion column
remapped y_true label
```

The benchmark is invalid if it does anything like:

```python
true_label = folder_name
pred_label = folder_name
```

or:

```python
pred_label = map_fer_label_to_aria_label(true_label)
```

The prediction must come from actual image inference, for example:

```text
image file -> vision/emotion model -> predicted label
```

Required debug output:

```python
print("VISION num_samples:", len(y_true))
print("VISION unique y_true:", sorted(set(y_true)))
print("VISION unique y_pred:", sorted(set(y_pred)))
print("VISION first 20 y_true:", y_true[:20])
print("VISION first 20 y_pred:", y_pred[:20])
```

Also print:

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, zero_division=0))
```

Additional checks:

```text
- Confirm images are actually loaded with cv2/PIL.
- Confirm DeepFace/emotion analyzer is actually called.
- Confirm failed image reads are not silently skipped in a way that leaves only easy samples.
- Confirm evaluation is not using only 10-20 samples.
- Confirm class mapping does not collapse the dataset into an overly easy binary problem.
```

---

## 2. Audio / RAVDESS Leakage Checks

RAVDESS filenames encode emotion labels. This is useful for `y_true`, but dangerous if accidentally used for `y_pred`.

Check whether the prediction is accidentally created from:

```text
filename
emotion code in filename
folder name
ground-truth label
same mapping used for y_true and y_pred
```

The benchmark is invalid if it does anything like:

```python
emotion_code = filename.split("-")[2]
y_true = map_ravdess_code(emotion_code)
y_pred = map_ravdess_code(emotion_code)
```

Prediction must come from actual audio inference:

```text
wav file -> prosody extractor/model -> predicted label
```

Required debug output:

```python
print("AUDIO num_samples:", len(y_true))
print("AUDIO unique y_true:", sorted(set(y_true)))
print("AUDIO unique y_pred:", sorted(set(y_pred)))
print("AUDIO first 20 y_true:", y_true[:20])
print("AUDIO first 20 y_pred:", y_pred[:20])
```

Also print confusion matrix and classification report.

Additional checks:

```text
- Confirm `.wav` files are actually loaded.
- Confirm prosody features are extracted from audio.
- Confirm prediction is not filename-derived.
- Confirm all RAVDESS emotion classes are represented.
- Confirm the evaluation is not using actor overlap in an unfair train/test setup.
```

---

## 3. Semantic / Mohler Checks

The Mohler result looks more realistic, but recall is low.

Check:

```text
Precision: 0.7331
Recall:    0.5230
F1-score:  0.5709
Accuracy:  0.7840
```

Diagnosis:

```text
The model may be conservative.
It predicts positive/correct answers only when very confident.
It misses many true positive cases.
```

Required checks:

```text
- What threshold converts score -> class?
- Is the class distribution imbalanced?
- Are scores being rounded too aggressively?
- Is the benchmark using binary classification or regression buckets?
- Are question-answer pairs correctly aligned?
```

Required debug output:

```python
print("TEXT num_samples:", len(y_true))
print("TEXT unique y_true:", sorted(set(y_true)))
print("TEXT unique y_pred:", sorted(set(y_pred)))
print("TEXT first 20 y_true:", y_true[:20])
print("TEXT first 20 y_pred:", y_pred[:20])
```

Also print confusion matrix and classification report.

Potential improvements after validation:

```text
- Tune classification threshold.
- Use rubric-aware grading.
- Use better sentence embeddings.
- Evaluate as regression using MAE/RMSE/Spearman in addition to classification.
```

---

## 4. Fusion Evaluation Leakage Checks

The perfect fusion result is the biggest red flag:

```text
ARIA Multimodal Attention Fusion accuracy = 1.0000
```

Module 4 fusion should normally output:

```text
fused_vector
modality_weights
modality_mask
modality_confidences
cross_modal_dissonance
```

It is not automatically a classifier unless a classifier/head is added after `fused_vector`.

Check exactly how `y_pred` is produced.

The benchmark is invalid if prediction is derived from fields that already encode the label, for example:

```text
competency_beginner_prob
competency_mid_prob
competency_expert_prob
vision_emotion_*_prob
audio_emotion_*_prob
```

and then compared to the same label that created those probabilities.

Potential leakage examples:

```python
# Bad if these probabilities were created directly from true label:
pred = argmax([
    competency_beginner_prob,
    competency_mid_prob,
    competency_expert_prob,
])
```

```python
# Bad:
fusion_label = true_label
```

```python
# Bad:
fusion_pred = label_mapping[y_true]
```

Required checks:

```text
- What is the fusion task label?
- Is the label binary, multiclass, emotion, competency, deception, or synthetic?
- How many fusion samples are used?
- Is there a trained classifier after fused_vector?
- If no classifier exists, what rule converts fused_vector -> y_pred?
- Does the fused_vector include direct ground-truth label probabilities?
```

Required debug output:

```python
print("FUSION num_samples:", len(y_true))
print("FUSION unique y_true:", sorted(set(y_true)))
print("FUSION unique y_pred:", sorted(set(y_pred)))
print("FUSION first 20 y_true:", y_true[:20])
print("FUSION first 20 y_pred:", y_pred[:20])
```

Also print confusion matrix and classification report.

---

# Required Benchmark Additions

Add a helper function to the benchmark script:

```python
def debug_eval(name, y_true, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    from collections import Counter

    print(f"\n=== {name} DEBUG ===")
    print("num_samples:", len(y_true))
    print("y_true counts:", Counter(y_true))
    print("y_pred counts:", Counter(y_pred))
    print("first 20 y_true:", list(y_true)[:20])
    print("first 20 y_pred:", list(y_pred)[:20])
    print("confusion_matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("classification_report:")
    print(classification_report(y_true, y_pred, zero_division=0))
```

Call this after every modality evaluation:

```python
debug_eval("VISION FER2013", y_true_vision, y_pred_vision)
debug_eval("AUDIO RAVDESS", y_true_audio, y_pred_audio)
debug_eval("TEXT MOHLER", y_true_text, y_pred_text)
debug_eval("FUSION ATTENTION", y_true_fusion, y_pred_fusion)
debug_eval("FUSION CONCAT", y_true_concat, y_pred_concat)
```

---

# Required Validity Criteria

Do not claim benchmark accuracy is valid until all of these are true:

```text
1. Sample count is printed.
2. Class distribution is printed.
3. Confusion matrix is printed.
4. y_pred is proven to come from model inference, not label metadata.
5. Train/test split is independent.
6. No file path, folder name, filename label, or annotation value is used in y_pred.
7. Fusion predictions are produced by a legitimate downstream classifier or clearly stated heuristic.
8. The same label source is not used to create both input probability features and target labels.
```

---

# Expected Diagnosis Outcome

Most likely:

```text
The current Vision and Audio results are inflated due to label leakage or overly simplified prediction logic.
The Fusion result is very likely invalid or synthetic because it reports perfect 1.0000 scores.
The Semantic result is the most realistic and should be improved after benchmark validation.
```

---

# Correct Next Task Order

## Step 1 — Validate Benchmark

Before model improvements, fix the benchmark script.

Target file:

```text
tests/benchmarks/run_baseline_benchmarks.py
```

Tasks:

```text
- Add debug_eval().
- Print sample counts and class distributions.
- Print confusion matrices.
- Confirm y_pred construction for each modality.
- Check if fusion task has a real classifier.
```

## Step 2 — Fix Any Leakage

If leakage is found:

```text
- Separate y_true construction from y_pred construction.
- Make y_pred come only from model inference.
- Do not use filename/folder labels except for y_true.
- Do not use true labels to create model output probabilities.
```

## Step 3 — Re-run Benchmarks

Re-run:

```powershell
python -m tests.benchmarks.run_baseline_benchmarks
```

or:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python .\\tests\\benchmarks\\run_baseline_benchmarks.py
```

## Step 4 — Only Then Improve Accuracy

After the benchmark is verified:

```text
1. Improve semantic recall.
2. Add WavLM/Wav2Vec2 embeddings for prosody.
3. Add temporal AU sequence features for vision.
4. Add temporal cross-attention for fusion.
```

---

# Coding Assistant Instruction

Please inspect the benchmark code first.  
Do not optimize the models yet.

The goal is to answer:

```text
Are these reported metrics valid, or are they inflated by leakage / small samples / synthetic labels?
```

Return:

```text
1. Exact source of y_true for each benchmark.
2. Exact source of y_pred for each benchmark.
3. Sample count and class distribution.
4. Whether each benchmark is valid or invalid.
5. Specific code changes needed to make the benchmark trustworthy.
```
