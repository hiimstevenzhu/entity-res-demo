# Helpers

This directory contains reusable, factory‑based utility packages that support the entity resolution workflows in this repository.

## Subpackages

- [`em_pipeline/`](./em_pipeline/README.md) – A modular Splink pipeline with factory patterns for data loading, comparisons, blocking rules, training, prediction, and visualisation.
- [`aggregation/`](./aggregation/README.md) – A behavioural‑aggregation framework that lets you define column‑wise aggregation logic via pluggable processors.

## Stand‑alone Modules

- `entity_resolution_aggregation.py` – Entry point that demonstrates how to use the aggregation framework on datasets produced by the data generator.
- `visualization_helper.py` – Common plotting functions (e.g., match‑weight histograms, uncatered charts) used across notebooks.

## Factory Pattern Overview

Each helper subpackage follows a similar factory‑registry pattern:

1. **Abstract Base Class** – Defines the interface (e.g., `ComparisonBase`, `BlockingRuleBase`, `ColumnProcessor`).
2. **Registry** – A dictionary (often populated via decorators) that maps string keys to concrete implementations.
3. **Factory Functions** – Simple look‑up helpers (e.g., `ComparisonRegistry.get(key)`) that return an instantiable class.

### Extending Without Touching Core Logic

To add new functionality:
1. Implement a class that inherits from the appropriate base class.
2. Register it in the subpackage’s `registry.py` (or equivalent) using the provided decorator or by manually inserting into the `_MAP` dictionary.
3. Use the new component by referencing its registry key in configuration YAMLs or factory calls.

Because the core orchestration code only interacts with the abstract base and the registry, **no changes to existing logic are required**.

## How to Use in Notebooks

See the example notebook `notebooks/splink_local_demo.ipynb` for end‑to‑end usage:
- Data loading via `helpers.em_pipeline.data_loader`
- Comparison and blocking rule creation via `helpers.em_pipeline.comparisons` and `.blocking_rules`
- Running training/prediction via `helpers.em_pipeline.training_pipeline` and `.prediction_pipeline`
- Visualisation via `helpers.visualization_helper` and `helpers.em_pipeline.visualization_pipeline`
- Aggregation via `helpers.aggregation` framework (see its README for details)

## Configuration

Most helper components are configured through YAML files that specify which factory keys to use and their parameters. For example, in `em_pipeline` you might have:
```yaml
comparisons:
  - name: "numerical"
    column: "amount"
    thresholds: [0.05, 0.15, 0.4, 0.7]
  - name: "datetime"
    column: "timestamp"
    day_thresholds: [0, 1, 2, 15]
```
Adjust these YAMLs (or the config passed to factory functions) to adapt the pipeline to new datasets without code changes.