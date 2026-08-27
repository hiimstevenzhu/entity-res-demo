Blueprint: Incremental Entity Resolution Data Generator

1. System Architecture Overview
The generator uses a Factory Pattern to separate the orchestration of entity generation from the generation of specific data types. It works in two phases:
- True Entity Space Construction: Generating a base set of real-world "Ground Truth" entities.
- Dataset Extraction & Flaw Injection: Simulating how different data sources (e.g., Bank X vs. Merchant) observe these entities with noise, missing values, and varying identifiers.       

[ Entity Factory ] (Orchestrator)
               |
      +--------+--------+

      |                 |
[Source A Gen]    [Source B Gen]  <-- Abstract Data Type Factories

      |                 |              (Categorical, Numerical, String)
[Dataset A]       [Dataset B]
      \                 /
    [Ground Truth Linkage Map] (5-15% Overlap)


2. Core Components & Factory Design
2.1 The Data Type Factory (DataTypeGenerator)
An abstract interface or base class that defines how specific data types are generated and corrupted.
- CategoricalGenerator:
Generates clean, deterministic matching categories (e.g., Account Status, Country Code).
Flaw Rule: No noise allowed, but supports a toggle for missing_probability.
- NumericalGenerator:
Generates values like Transaction Amount, Age, or Balance.
Flaw Rule: Supports two modes via configuration: Clean (exact matches) or Noisy (applies Gaussian/random noise or rounding shifts).
- DateTimeGenerator:
Generates base event timestamps (e.g., transaction execution time).Flaw Rule: Simulates system logging delays or clock unsynchronisation across systems. Applies a Gaussian noise distribution with a standard deviation (σ) of 2 hours to the timestamp. Supports missing_probability.
- StringGenerator (KIV):
Placeholder interface for text (Names, Addresses) to introduce typos or phonetic noise later.

2.2 The Entity Profile Manager (EntitySpace)
Before generating rows, we must define the hidden "real-world" entities to ensure we control the exact match rate.
- Generates \(N\) unique global entity IDs.
- Determines which entities are shared between Dataset A and Dataset B to strictly enforce the 5–15% aggregate match rate.
- Maps local source-specific unique identifiers to the global entity ID (simulating non-corresponding unique IDs).

3. Mathematical & Logic Configurations
3.1 Flaw & Noise Injection Formulas
- Missing Data (All Types):
\(P(\text{Value}=\text{NaN})=p_{\text{missing}}\)
- Numerical Noise:
\(\text{Observed\ Value}=\text{True\ Value}+\mathcal{N}(0,\sigma ^{2})\)
(or a percentage-based variance for transaction amounts).
- DateTime Temporal Noise:
\(\text{Observed\ Time}=\text{True\ Time}+\Delta t\)
Where Δ t is drawn from a normal distribution converted to time units:
\(\Delta t\sim \mathcal{N}(\mu =0\text{\ hours},\sigma =2\text{\ hours})\)

3.2 Scalability Considerations (Millions of Rows)
To prevent memory exhaustion when scaling to millions of rows, the Python implementation must avoid nested loops:
- We use Vectorised Operations via NumPy/Pandas.
- Generate base arrays in block memory chunks.
- Apply noise using boolean masks rather than row-by-row iteration.
- Utilise pd.to_timedelta combined with np.random.normal to vectorise the generation of noisy temporal offsets across millions of rows instantly.


4. Usage

Configuration is done by a structured dictionary that can be tweaked from the config.yaml file.
