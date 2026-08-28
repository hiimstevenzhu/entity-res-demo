set comparisons

Categorical

\text{Jaccard}(L, R) = \frac{|L \cap R|}{|L \cup R|} # set jaccard

\text{MagCosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\max(|\mathbf{u}|^2, |\mathbf{v}|^2)} # mag-normalised cosine



Text Based

\text{SoftJaccard}(L, R) = \frac{\sum_{x \in L} \max_{y \in R} \text{JW}(x, y) + \sum_{y \in R} \max_{x \in L} \text{JW}(x, y)}{2 \cdot |L \cup R|}

\text{MongeElkan}(L, R) = \frac{1}{|L|} \sum_{x \in L} \max_{y \in R} \text{JW}(x, y)

\text{SoftDice}(L, R) = \frac{\sum_{x \in L} \max_{y \in R} \text{JW}(x, y) + \sum_{y \in R} \max_{x \in L} \text{JW}(x, y)}{|L| + |R|}


Usage:

\section*{Unified Soft String-Matching Metrics Framework}




\subsection*{1. Definitions and Notation}
Let  and  be two token lists representing the left and right records, respectively.
Let  and  denote the total number of token slots in each list.
Let  represent the Jaro-Winkler similarity score between token  and token .

We define the directional \textbf{Best-Match Sums} from left-to-right ($S_{L \to R}$) and right-to-left ($S_{R \to L}$) as follows:
\begin{equation}
S_{L \to R} = \sum_{x \in L} \max_{y \in R} \text{JW}(x, y)
\end{equation}
\begin{equation}
S_{R \to L} = \sum_{y \in R} \max_{x \in L} \text{JW}(x, y)
\end{equation}

\subsection*{2. The Four Soft Metrics}

\subsubsection*{Monge-Elkan Similarity (Asymmetric)}
Measures the average of the best-matching scores from the query set ($L$) to the reference set ($R$):
\begin{equation}
\text{MongeElkan}(L, R) = \frac{S_{L \to R}}{|L|}
\end{equation}

\subsubsection*{Symmetric Soft Jaccard Similarity}
Normalizes the combined mutual best-match scores by the total size of the distinct token union:
\begin{equation}
\text{SoftJaccard}(L, R) = \frac{S_{L \to R} + S_{R \to L}}{2 \cdot |L \cup R|_{\text{distinct}}}
\end{equation}

\subsubsection*{Symmetric Soft Dice Coefficient}
Normalizes the combined mutual best-match scores by the total raw token slots available across both lists:
\begin{equation}
\text{SoftDice}(L, R) = \frac{S_{L \to R} + S_{R \to L}}{|L| + |R|}
\end{equation}

\subsubsection*{Soft Simpson Coefficient (Overlap Metric)}
Normalizes the average mutual best-match score by the size of the smaller token list, isolating it from catalog size imbalances:
\begin{equation}
\text{SoftSimpson}(L, R) = \frac{S_{L \to R} + S_{R \to L}}{2 \cdot \min(|L|, |R|)}
\end{equation}

\subsection*{3. Unified Binned Comparison Logic}
For any given metric score , the assignment to a Splink comparison level is generalized using the following case structure:

\begin{equation}
\text{ComparisonLevel}(S) =
\begin{cases}
\text{Null / Empty} & \text{if } L \text{ is Null, } R \text{ is Null, } |L|=0, \text{ or } |R|=0 \
\text{High Similarity} & \text{if } S \ge \tau_{\text{high}} \
\text{Medium Similarity} & \text{if } S \ge \tau_{\text{med}} \
\text{Low Similarity} & \text{if } S \ge \tau_{\text{low}} \
\text{Else} & \text{otherwise}
\end{cases}
\end{equation}

\noindent where the operational mapping thresholds  are defined as:

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Metric} &  &  &  \ \hline
Monge-Elkan     & 0.85                  & 0.70                 & 0.55                 \
Soft Jaccard    & 0.85                  & 0.65                 & 0.40                 \
Soft Dice       & 0.85                  & 0.65                 & 0.40                 \
Soft Simpson    & 0.85                  & 0.65                 & 0.40                 \ \hline
\end{tabular}
\caption{Operational Level Thresholds}
\end{table}







Numerical







