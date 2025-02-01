# ADLSys Coursework

## Lab 0: Introduction to Mase

### Tutorial 1

### Tutorial 2

**Task:** Remove the `attention_mask` and `labels` arguments from the `hf_input_names` list and re-run the following cell. Use `mg.draw()` to visualize the graph in each case. Can you see any changes in the graph topology? Can you explain why this happens?

**Answer:** The mase graph has replaced attention_mask node with a identity mapping node, essentially removed the attention mask. The labels also disappeared.

## Lab 1: Model Compression (Quantization and Pruning)

### Tutorial 3

**Task:**  In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32.

Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3.

Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.

**Answer:**
PTQ results: [0.5, 0.8162, 0.8364, 0.83552]
QAT results: [0.5, 0.8398, 0.8428, 0.84236]


### Tutorial 4

**Task:** Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9.

Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4.

Plot separate curves for Random and L1-Norm methods to evaluate the effect of different pruning strategies.

**Answer:** 


## Lab 2: Neural Architecture Search

### Tutorial 5
