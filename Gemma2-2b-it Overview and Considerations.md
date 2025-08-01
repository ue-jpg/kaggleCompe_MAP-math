# Final Model Overview: Gemma-2-2b-it with Improved Prompts + QLoRA
**Date**: July 24, 2025

## 🤔 Analysis and Considerations

### Performance Comparison with Other Models
Despite using a significantly larger model (Gemma-2-2b-it with ~2.6B parameters), the performance improvement compared to smaller models is limited. For example, DeBERTa-v3-xsmall, which has approximately 1/100th the parameters, achieves MAP@3 scores around 0.93. This suggests that model size alone is not the primary factor limiting performance in this task.

### Performance Ceiling Observations
Examining the leaderboard, most submissions appear to plateau around 0.94 MAP@3 score, indicating a potential performance ceiling for this competition. This plateau suggests fundamental limitations that are not easily overcome by simply scaling model size or improving training techniques.

**Resource Constraints and Scalability**: It's important to note that MAP@3 scores don't necessarily improve linearly with model parameter count. If the goal were to improve scores simply by scaling up model size, it would require significantly larger computational resources and associated costs, which are not feasible in my current environment.

### Potential Underlying Issues

#### Misconception Category Problems
The current labeling scheme may be working against optimal performance. The Category:Misconception format creates several challenges:

1. **Severe Class Imbalance**: Many misconception categories have only one or very few samples, making it nearly impossible for models to learn meaningful patterns for these rare classes.

2. **Non-generalizable Labels**: Some misconception categories are highly specific and may not represent patterns that generalize well to unseen data.

3. **Complex Multi-dimensional Classification**: The current format combines multiple classification dimensions (True/False for correctness + Correct/Misconception/Neither for understanding type) into a single label, potentially making the learning task unnecessarily complex.

#### Proposed Alternative Approaches

**Simplified Classification Approach:**
Focus on the core understanding classification (Correct/Misconception/Neither) while excluding the True/False correctness dimension and specific misconception categories. This would:
- Reduce class imbalance issues
- Focus on the most generalizable aspects of student understanding
- Allow better utilization of larger model capabilities

**Two-Stage Pipeline Approach:**
An alternative strategy could involve:
1. Using a high-performance pre-trained LLM to identify unclear points, errors, or misconceptions in student explanations
2. Applying a secondary classification model to categorize these identified issues

This approach could potentially leverage the superior reasoning capabilities of large language models while avoiding the current labeling complexity issues.

### Hypothesis
The fundamental issue may not be model capacity but rather the labeling framework itself. Since our Gemma-2-2b model and other participants' larger models should theoretically have superior baseline LLM performance, addressing the labeling complexity could unlock significantly better prediction accuracy.

**Note**: These observations are based on empirical results and leaderboard analysis, and represent hypotheses that would require further experimentation to validate.

---

## 🚀 Model Summary
**Model Name**: `gemma-2-2b-improved-prompts-qlora`
**Base Model**: google/gemma-2-2b-it (~2.6B parameters)
**Training Method**: QLoRA (Quantized Low-Rank Adaptation)
**Task**: 65-label classification for mathematical misconception detection

## 📊 Performance Results
- **Final MAP@3 Score**: **0.9411** (94.11%)
- **Final Accuracy**: 0.8894 (88.94%)
- **Evaluation Loss**: 0.3691
- **Training Epochs**: 3.0
- **Evaluation Runtime**: 473.05 seconds
- **Evaluation Speed**: 15.52 samples/second
- **Evaluation Steps per Second**: 1.94
- **Improvement from Baseline**: +1.1% (0.93 → 0.9411)

## 🔧 Technical Specifications

### QLoRA Configuration
- **Quantization**: 4-bit (nf4) with double quantization
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Memory Reduction**: ~75% compared to full fine-tuning

### Training Parameters
- **Epochs**: 3
- **Batch Size**: 8 (per device)
- **Gradient Accumulation**: 4 steps (effective batch size: 32)
- **Learning Rate**: 1e-4
- **Max Token Length**: 1024
- **Training Time**: ~6 hours on A100 GPU

### Training Environment
- **Platform**: Google Colab Pro
- **GPU**: NVIDIA A100 (40GB VRAM)
- **Compute Units Consumed**: 50 units (~$5.00 USD cost)
- **Code Backup**: Training code preserved in input section for reference

## 📝 Prompt Engineering Improvements

### Enhanced Prompt Structure
Based on `final_compact_prompt.py` with the following optimizations:

1. **Early Guidelines Placement**: Classification guidelines positioned at the beginning
2. **Complete Label Coverage**: All 65 labels including False_Correct:NA
3. **Clear Task Definition**: Explicit instruction for exact label selection
4. **Structured Format**: Organized question-answer-explanation flow

### Sample Prompt Template
```
You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: [Question Text]
Correct Answer: [MC_Answer]
Student's Explanation: [Student Explanation]

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

TASK: Classify this student's response using EXACTLY ONE of these 65 labels:
[Complete label list...]

Classification:
```

## 🎯 Key Improvements Over Baseline

### 1. Prompt Optimization
- **Token Efficiency**: ~741 tokens average (optimal for Gemma-2B)
- **Label Completeness**: Full 65-label taxonomy support
- **Context Structure**: Enhanced problem-solution-explanation flow

### 2. QLoRA Benefits
- **Memory Efficiency**: 75% reduction in GPU memory usage
- **Training Stability**: Improved convergence with 4-bit quantization
- **Parameter Efficiency**: Only ~0.5% of parameters trained (adapters)

### 3. Architecture Enhancements
- **Gradient Checkpointing**: Memory optimization for long sequences
- **Group Batching**: Efficient processing of variable-length inputs
- **Mixed Precision**: bf16 for QLoRA stability

## 🛠️ Development Environment

### AI-Assisted Development
- **Primary Tool**: GitHub Copilot for code generation and documentation
- **Language Support**: English documentation and code comments (**author is non-native English speaker**)
- **Code Coverage**: Majority of implementation assisted by Copilot, including model training, data processing, and submission notebooks

### Training Infrastructure
- **Cloud Platform**: Google Colab Pro with premium GPU access
- **Hardware**: NVIDIA A100 GPU (40GB VRAM) for efficient QLoRA training
- **Cost Management**: 50 compute units consumed (~$5.00 USD total cost)
- **Code Preservation**: Complete training codebase backed up in input section for reproducibility

---
**Model Created**: July 23, 2025
**Framework**: Transformers + PEFT + QLoRA
**Competition**: MAP - Charting Student Math Misunderstandings
