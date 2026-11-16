# EgoAgent reduced Mac pipeline — Evaluation results

## Task

Reduced future world state prediction on WalkingTours frames.

Input:

- current frame: frame_t
- future frame: frame_t+5 seconds

The model predicts the future visual embedding from the current visual embedding.

## Pipeline

1. Extracted WalkingTours frames at 1 FPS.
2. Built frame pairs: frame_t -> frame_t+5.
3. Stage 1: trained a contrastive visual encoder.
4. Stage 2: trained a future embedding predictor.
5. Evaluation: compared the model against a persistence baseline.

## Results

| Metric | Baseline | Model | Improvement |
|---|---:|---:|---:|
| Cosine similarity | 0.725701 | 0.731756 | +0.006055 |
| MSE | 0.002143 | 0.002096 | +0.000047 |

## Interpretation

The future predictor improves over the persistence baseline.  
The improvement is modest but measurable, which is expected because adjacent walking-tour frames are often visually similar.

The result confirms that the reduced pipeline is working:

- Stage 1 learns a visual embedding space.
- Stage 2 learns a temporal transformation from current state to future state.
- The model performs slightly better than simply reusing the current frame embedding as the future embedding.
