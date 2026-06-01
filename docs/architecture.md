# DLTPKT Architecture Notes

This document records the behavior of the current release code. It is not an
experiment log.

## Main Scope

The release package contains one model path:

- Model: `DLTPKT`
- Dataset: processed `Statics`
- Training splits: `train_all_feature.txt` and `test_all_feature.txt`
- Main loss: binary cross-entropy on the next-response prediction

## Static Structure

The skill layer and big-concept layer construct directed structural roles from
shared node embeddings. Each layer projects its shared embeddings into
predecessor and successor roles, then computes directed scores as:

```text
successor_roles @ predecessor_roles.T
```

The current implementation applies a sigmoid, a `0.6` hard threshold, and row
normalization. A straight-through estimator keeps the sparse binary graph in
the forward pass while allowing gradients to flow through the soft scores.

## Prediction Path

For the target question, DLTPKT uses an explicit hierarchical mastery path and
a separately inspectable residual-ability path:

```text
concept_mastery = sigmoid(concept_mastery_readout(concept_state))
concept_support_for_skill = skill_concept_attention @ concept_mastery
skill_mastery = sigmoid(skill_mastery_readout(skill_state) + beta * concept_support_for_skill)
target_skill_mastery = weighted_mean(skill_mastery, target_question_skills)
target_concept_mastery = weighted_mean(concept_mastery, target_question_concepts)
mastery_ability = linear([target_skill_mastery; target_concept_mastery])
residual_ability = MLP([fused_state; fused_question_embedding])
difficulty = linear(fused_question_embedding)
probability = sigmoid(5 * (mastery_ability + residual_ability - difficulty))
```

The node-level mastery readouts and their target-question summaries are bounded
to `(0, 1)`. The mastery contribution, residual ability, and difficulty are
returned separately for auditing. The residual path preserves predictive
capacity but means this is not a strict mastery-only bottleneck.

## Mastery Readout Interface

When called with `return_details=True`, the model additionally returns:

```text
skill_mastery
concept_mastery
target_skill_mastery
target_concept_mastery
```

The first two tensors contain node-level sigmoid readouts. The latter two are
target-question-specific weighted summaries.

The main BCE loss trains both readout layers because big-concept mastery feeds
skill mastery and both target mastery summaries contribute to the prediction.
These values are useful for case analysis, but should not be reported as
calibrated mastery probabilities unless their calibration has been separately
validated.

## Checkpoint Compatibility

`checkpoints/DLTPKT_statics_best.pth` is a legacy artifact. It uses an older
model structure and `10` big concepts, while the current training default is
`5`. Train the current code before running the default evaluation command.
