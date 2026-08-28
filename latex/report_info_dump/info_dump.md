1) Abstract: specific scope mentioning, what was used, results

2) Problem statement, context, scope

3) Framework, Theory…?

split into 2 portions: a) framework: local, global

b) theory for local: probabilistic matching, theory for global: aggregation and agreement

4) Data generation implementation - artificial data

5) splink usage explanation - syntax, blocking rules, coverage…? maybe things that the initial fellegi sunter didnt account for or something

blocking

fuzzy levels

“It would be good if you could include one or both of the following in your report:

1. A rationale behind the "strictest blocking rules" estimation method
2. A comparison on the final results when different approaches are used
  1. e.g. by varying the blocking rules, or
  2. by varying the estimated recall”

6) examples:

1: artificial dataset: initial runs split via 1a) local, 1b) global aggregation

2: artificial dataset: results varying different configs for 1a) local, 1b) global aggregation

3: public dataset: 1a) local, 1b) global aggregation \[NOTE: required to discuss estimates for precision and recall here: bounding\]

7) extremeties: cardinality - explanation for when and why splink + aggregation might not work + an initial proposal for a methodology to overcome it

0: methodology: collect as many orthogonal sim scores as possible, throw them into another round of probabilistic matching. adjust for correlation.

1: artificial dataset: low card results and combinatorial grain, initial exploration into a secondary layer of splink

8) Future work/considerations

1) mixture models for threshold selection

2) big work - there are papers exploring the specific topic of “whats the minimum number of labels” → seems to suggest 30 ish labels is sufficient.

This is a highly advanced, structurally sound plan for a data linking/entity resolution research paper using Splink.

To improve this for a formal academic or technical research report, we need to translate your current notes into standard academic sections (e.g., introduction, methodology, experiments) while keeping the exact logical flow you designed.

Here is the refined, publication-grade structure optimized for a clean markdown-to-LaTeX conversion later.

---

## Revised Research Report Blueprint

## 1\. Abstract \[STAGE 1\]

- Context & Scope: Brief statement on entity resolution scaling challenges.
- Methodology: Explicitly name the hybrid approach (Local Splink matching + Global Aggregation).
- Results: Summary of the key performance metrics achieved across the artificial and public datasets.

This report details a methodology to global entity resolution (defined in QR3), under a few assumptions. This report shows that, given that these assumptions hold, the methodologies hold \[recall and precision/metrics\]…?

What else:

\- show two different methods, with metrics on so and so.

MENTION SCOPE: 1) unsupervised learning, 2) specific important assumptions: stable identifiers

## 2\. Introduction \[core takeaways: the scope, and constraints given in the problem dictates the actual method used by a large amount\] \[STAGE 2\]

- Problem Statement:
- Context & Scope:
- Contributions: Map out the core value of your local/global framework.

\[skeleton - general definition and context\] ER: definition, starting from inconsistency in quality of data, and its relevance for data integration for use. Traditional approaches are labour intensive and difficult for various reason: attribute ambiguity, data entry errors, missing values, temporal effects on attributes etc. → ER is actually a wide scope of problem, on two fronts: the specific structure of data to uncover, and inference methods. → Give examples - a clustering problem with no labels has different constraints compared to a classification problem with [labels.](http://labels.As) → An extremely wide range of methods can be employed for ER (ANNEX: a summary of methods, inference styles, problem style)

\[skeleton - scoping\] Problem Statement is not as important as specifically defining the constraints given to the problem → This report thus details an approach to ER with the following constraints:

- End goal of matching global entities
- Unsupervised learning \[scopes it down to FS/graphical models w dependencies etc/MCMC or clustering methods\]
- Minimise some sort of total cost…?
- Using textual, numerical, datetime columns ==\[scopes out of using LLM methods for semantic matching\], also \[scopes out of graph based networks where the core concept is in behavior-heavy matching\]==
  - diff between the approach here is that if we swapped our datasets for something that is more transactional in nature - capturing behaviour or network, then graphical approaches would be better.
- impt: identifiers defining global entities can differ between datasets, but are guaranteed to be consistent/stable within the dataset. \[this scopes us down away from unnecessary compute in clustering/graph networks\]

\[skeleton - methodology and results/contributions\] Given the specific constraints, the best (if not only) approach to solving this problem is to use FS → introduce splink, explain two frameworks for exploration: local into aggregate for global, and aggregate behaviour into global → introduce the recall and precision across 1) artificial data, 2) acm dataset, 3) music-voters.

\[skeleton - structure, and focus\] Report will rotate focus between two core tenets: rationale in decisions and approach to practical implementation

Entity resolution encapsulates a wide range of problems in relation to record linkage.

## 3\. Theoretical Framework & Methodology \[what are we really dealing with/ how do we build an intuition behind what we’re looking at\] \[STAGE 1\]

- 3.1 Local vs. Global Entities: High-level framework overview with analogy,
  - Just show the diagram, show an example. \[we already have it, just let me insert it\]

- 3.2 a) theory: Local Layer (Probabilistic Matching): Mathematical foundation of Fellegi-Sunter parameters used in local blocks.
  - assumptions required
  - method summary w u-prpb, m-prob, prior and their estimation w EM (give math under annex)
  - resulting output - odds, weight, proba.
  - Show it gives minima in the amount of labour required for checks or something
  - gaps:
    - core problem is the assumption and how its realistically violated every time, but show that this assumption is largely, in practice, able to be relaxed to a large extent.
    - the core uncertainty in the effect of initial estimation parameters and their effect on the results. importantly, two estimations are might be required: initial estimation recall, as well as wtv
- 3.3 b) Implementation: Splink - implementation of FS with expectation maximisation
  - essentially an implementation of fellegi sunter with EM using sql expressions for comparisons in order to parallelise compute using cpu by exploiting whatever parallisable structure.
  - estimates values while utilising blocking rules to drastically reduce the number of pairs to look at. (explain that record matching is a problem that scales in R² for both estimation and resulting classifcation, and blocking rules allows for a large reduction in the number of pairs to review)
  - whatever issues that might be resolved in the original paper that splink might fix

- 3.3 Global Layer (Aggregation & Agreement): aggregation functions: naive, noisy_or, expected_score, etc. give the math equations and unify their definitions

## 4\. System Implementation & Tooling \[STAGE 1\]

- 4.1 Data Generation: Rules and synthetic corruptions used to build the artificial dataset. \[under data_generator\]
- 4.2 Splink Implementation: Technical configurations (Syntax, custom comparison levels, and fuzzy logic features over standard Fellegi-Sunter).
- 4.3 Blocking Strategy & Recall Bounds:
  - _The "Strictest Blocking Rules" Estimation Rationale_: Mathematical justification for your chosen baseline filter.
  - _Coverage Analytics_: How you track pairs missed by the initial blocking configurations.

4.1 Data Generation:

\[skeleton\] formalise the data generation method first

\[skeleton\] introduce actual generation implementation (also add code into annex)

\[image\]…? fuh.

4.2 Splink Implementation:

\[skeleton\]

## 5\. Experimental Results & Discussion

- 5.1 Experiment 1a: Artificial Baseline: Initial runs benchmarking Local matching into Global Aggregation. (splink_process, EM_smallest, small, and then splink_agg in global_notebooks)
- 5.3 Experiment 2: Sensitivity Analysis:
  - Performance impacts when varying blocking rules and altering target recall bounds. (TODO())
  - Varying different hyperparams on data generation and exploring this variance with the splink performance (splink_c2, o50, low_card)
- 5.1 Experiment 1b: Public Dataset Validation: Real-world performance analysis featuring precision/recall bounding estimations. \[DBLP ACM and Music Voters\] (splink_music in local and splink_music_agg in global)
- 5.2 Experiment 3a:”Behavioural Aggregation (agg_eg, and agg_eg_normal.py)

## 6\. Edge Cases & Structural Extremities (Cardinality)

- 6.1 Failure Modes: Structural breakdown analysis explaining exactly why Splink + aggregation fails at low-cardinality/high-combinatorial extremes. (splink_low_card)
- 6.2 Proposed Mitigation (Layer 2 Processing): Methodology for collecting orthogonal similarity scores. (splink low_card, low_card copy?)
- 6.3 Multi-Stage Probabilistic Tuning: Secondary Splink processing layer adjusted for feature correlation. (splink low_card, low_card copy?)
- 6.4 Preliminary Edge Case Results: Initial testing of the layer 2 setup on low-cardinality subsets.

## 7\. Discussion or something

1) answer the specific question of if confidence scores are necessary

a) no: why is it not necessary → we show that current methods are actually more than reliable (empirical), confidence scores increase complexity (explanability), and that the confidence scores also are mathematically illogical/redundant (logic or sumn).

2) whatever else to discuss

## 8\. Future Work & Conclusion

- 7.1 Automated Thresholding: Implementing mixture models to dynamically pick optimal match thresholds.
- 7.2 Active Learning & Label Minimisation: Exploring recent literature benchmarks regarding the "30-label minimum" requirement for human-in-the-loop validation.

## 9\. Annex:

- 8.1 Summary of ER methodologies: method, inference method, etc.
- 

---

## 💡 The "Lazy LaTeX Conversion" Pro-Tips for this Layout

Since you are writing this in Confluence first and using an LLM to generate the LaTeX later, follow these rules to ensure zero-click conversion:

- Prefix your section names: Use markdown headers exactly matching the hierarchy above (`# 3. Theoretical Framework`, `## 3.1 Local vs. Global Architecture`). The LLM will effortlessly parse these into `\section{}` and `\subsection{}`.
- Draft your tables cleanly in Section 5: Do not merge any cells in your experiment summary tables. Let the rows and columns stay flat so the LLM can instantly map them to LaTeX `tabular` environments.
- Tag your data visuals: Under Section 5 and 6, type `[INSERT FIGURE: artificial_baseline_chart.png]` right below your captions. When you move to your offline editor, you just drop the images into your project folder under those exact names.

If you like this structure, tell me: Do you want to map out the exact mathematical variables for Section 3 right now, or should we script a sample code block for how Section 4's Splink syntax should be documented in your notes?

## 0\. Abstract

## 1\. Problem Statement

we can probably reference what jess wrote in qr1,2,3 and then play by ear for whatever else man

## 2\. Approach

### 2a. Framework

### 2b. Theory
