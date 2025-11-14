# Two Moons FSP Classification Example

This example demonstrates binary classification on the two moons dataset using FSP (Function-Space Prior) Laplace approximation.

## What This Example Shows

1. **Training a neural network** on the two moons dataset
2. **Computing FSP posterior** using unstructured kernel
3. **Quantifying uncertainty** in predictions
4. **Visualizing** decision boundaries with uncertainty

## Running the Example

```bash
python examples/two_moons_fsp_classification.py
```

## Requirements

```bash
pip install jax jaxlib numpy matplotlib scikit-learn torch
```

## Output

The script will:
1. Generate and train on the two moons dataset
2. Select context points for FSP
3. Compute the FSP Laplace posterior
4. Generate predictions with uncertainty estimates
5. Save a visualization to `two_moons_fsp_laplace.png`

## Key Concepts

### FSP Laplace Approximation

The FSP approach:
- Uses a **function-space prior** (GP prior on network outputs)
- Computes a **Laplace approximation** at the MAP estimate
- Provides **uncertainty quantification** through the posterior

### Kernel Structure

This example uses `KernelStructure.NONE` (unstructured kernel):
```python
from laplax.curv import KernelStructure, create_fsp_posterior

posterior = create_fsp_posterior(
    model_fn=mlp_forward,
    params=trained_params,
    x_context=x_context,
    kernel_structure=KernelStructure.NONE,  # Unstructured kernel
    kernel=kernel_fn,
    prior_variance=prior_variance,
    n_chunks=2,
    max_iter=50,
)
```

### Uncertainty Visualization

The example creates three plots:
1. **Mean predictions**: Decision boundary from the trained model
2. **Predictive uncertainty**: Standard deviation across posterior samples
3. **Model confidence**: Areas where the model is most certain

## Customization

You can modify:
- **Network architecture**: Change `layer_sizes` in `init_mlp_params`
- **Kernel parameters**: Adjust `lengthscale` and `variance` in `rbf_kernel`
- **Number of context points**: Change `n_context_points` in `select_context_points`
- **Context selection strategy**: Try "sobol", "halton", or "latin_hypercube"

## Expected Output

```
======================================================================
Two Moons Classification with FSP Laplace
======================================================================

1. Generating two moons dataset...
   Dataset shape: X=(300, 2), y=(300,)

2. Training MLP...
   Epoch 20/100, Loss: 0.XXXX
   Epoch 40/100, Loss: 0.XXXX
   ...
   Training accuracy: XX.XX%

3. Selecting context points...
   Selected 50 context points

4. Computing FSP posterior...
   Posterior rank: XX

5. Making predictions with uncertainty...
   Mean probability range: [X.XXX, X.XXX]
   Std range: [X.XXX, X.XXX]

6. Creating visualizations...
   Saved visualization to 'two_moons_fsp_laplace.png'

======================================================================
Done! FSP Laplace successfully applied to two moons classification.
======================================================================
```

## Related Examples

For more FSP examples, see:
- `0004_fsp_laplax.ipynb` - FSP on regression tasks
- `0004_function_space_laplace.ipynb` - Detailed FSP walkthrough
