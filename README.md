# Konwledge-embedded_Batch_BO
This work proposes a knowledge-embedded batch Bayesian optimization (BO) framework tailored for the laser fabrication of resistive strain gauges. In this approach, laser processing is employed to induce localized carbonization on polymer substrates (such as PEEK), forming sensing elements that respond to local strain.

The performance of these strain-sensitive elements, specifically their initial resistance and gauge factor (GF), is governed by a multi-dimensional laser parameter space, including laser power, defocus distance, repetition rate, and scanning speed. However, this design space is typically vast and non-convex, and each fabrication condition requires expensive and time-consuming physical experiments. In particular, when the resistive element serves as the strain-sensing unit of an LCR resonant wireless sensor, evaluating the effectiveness of each laser condition entails the complete fabrication and characterization of the entire device, further elevating experimental costs. Consequently, the problem can be formulated as a high-cost black-box optimization task, where the objective is to efficiently identify optimal laser parameters using as few evaluations as possible. Bayesian optimization is well-suited to such scenarios due to its ability to model uncertainty and balance exploration with exploitation.

To improve the practicality and convergence efficiency of the optimization, this framework integrates prior domain knowledge from laser-material interactions into the modeling process. The final optimization goal is to obtain a resistive strain gauge that simultaneously exhibits low baseline resistance (for signal-to-noise ratio enhancement) and a high gauge factor (for sensitivity improvement), thereby enabling high-performance wireless sensing in biomedical or structural monitoring applications.
# Packages
The following libraries are necessary for running the codes.
```
botorch==0.6.0
gpytorch==1.6.0
joblib==1.4.0
matplotlib==3.5.1
numpy==1.22.4
pandas==1.3.5
scikit_learn==1.0.1
scikit_learn_extra==0.3.0
scipy==1.7.3
torch==1.10.0
tqdm==4.62.3
umap_learn==0.5.7
```
Install requirements using below command.
```
pip install -r requirements.txt
```
# Pipeline
The proposed optimization pipeline consists of two main phases encompassing three sequential steps, designed to obtain strain-sensitive resistors with both low baseline resistance and high gauge factor (GF). The first phase focuses on minimizing the baseline resistance, while the second phase performs constrained optimization of GF under a resistance constraint. Notably, the experimental cost of the second phase is significantly higher than that of the first.
## Phase I: Resistance Optimization via Knowledge-Embedded Batch BO
### Step 1: Initial Sampling
A Latin Hypercube Sampling (LHS) strategy is employed to perform stratified sampling across the laser parameter space (e.g., laser power, defocus distance, repetition rate, scan speed), generating an informative and representative set of initial experiments.
```
python step1_init.py
```
### Step 2: Iterative Resistance Optimization
A knowledge-embedded batch BO algorithm is used to guide the search for laser conditions that minimize the linear resistance of the strain gauge. Each execution of the BO script corresponds to one optimization iteration, during which a batch of potentially optimal laser conditions is recommended for physical fabrication and experimental validation. The observation dataset is then updated with the new results, and this loop continues until a predefined stopping criterion (e.g., convergence or resource limit) is met.
```
# interative
python step2_Ropt.py
```
## Phase II: Gauge Factor Optimization under Resistance Constraint
### Step 3: Iterative Constrained GF Optimization
Upon completion of the resistance optimization phase, a Gaussian Process (GP) surrogate model is constructed to capture the distribution of resistance across the laser parameter space. This GP model serves as the cost constraint in the subsequent Constrained Expected Improvement (CEI)-based GF optimization. A second BO procedure is then performed to explore laser conditions that maximize the GF while satisfying the resistance constraint inferred from the GP model.
```
# interative
python step3_GFopt.py
```
Through this two-phase framework, the pipeline effectively identifies laser processing conditions that yield strain-sensitive resistors with low resistance and high gauge factor, achieving both signal quality and sensitivity requirements for downstream sensor applications.
# Data
Data needed for each phase is in the folder ***./data***
# Tools
These scripts also support the visualization of the optimization process, including the convergence behavior of the Gaussian Process (GP) models, as well as the embedding and visualization of high-dimensional laser parameter spaces. Such visual analytics facilitate intuitive understanding of model performance and decision boundaries, thereby enhancing the interpretability and traceability of the optimization procedure. Details can be seen in the folder ***./visualization and tools***
