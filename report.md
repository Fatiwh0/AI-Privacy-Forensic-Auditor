\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage{xcolor}
\geometry{margin=2.3cm}

\title{AI Privacy Forensic Auditor\\Living Scientific Report}
\author{Team Project - Data Mining}
\date{Last update: 2026-02-22 (Expanded beginner-level narrative)}

\begin{document}
\maketitle

\section{Final End Goal (The Big Picture)}
The final project objective is to build a complete \textbf{AI Privacy Forensic Auditor} that can:
\begin{enumerate}
  \item train and analyze a baseline predictive model,
  \item detect whether private training membership can be inferred (\textbf{MIA}),
  \item evaluate defenses and privacy-utility tradeoffs,
  \item produce reproducible scientific evidence (tables, figures, logs),
  \item and satisfy class rubric requirements with a clear written and oral narrative.
\end{enumerate}

In simple terms: we are not only building a classifier.
We are building an \textbf{audit system} that asks:
\textit{``Can an attacker guess if someone was in the training data?''}

\section{Where We Are Right Now}
\subsection{Current Progress}
\begin{itemize}
  \item Completed pipeline stages: \texttt{ingestion}, \texttt{preprocessing}, \texttt{eda}, \texttt{reliability}, \texttt{mia\_shadow}
  \item Current status: core baseline and leakage detection are working
  \item Remaining major stages: defense, association rules, clustering, dashboard polishing, final report/PPT integration
\end{itemize}

\subsection{Why This Matters}
At this point, the project already has:
\begin{itemize}
  \item reproducible data pipeline,
  \item scientific exploratory analysis,
  \item reliability evidence (not just one run),
  \item initial privacy leakage quantification through MIA.
\end{itemize}
This is a strong baseline for both grading and team continuation.

\section{Project Story Chronology (Step by Step)}
\subsection{Step 0: Scope Alignment and Constraints}
\textbf{What was done:}
The project scope was aligned to the professor rubric and the 2-week deadline, with class-first priority and portfolio-quality structure.

\textbf{Why:}
Without scope control, teams lose points by building technically interesting but rubric-incomplete work.

\textbf{Goal:}
Guarantee 18+ potential by mapping every rubric criterion to concrete artifacts.

\textbf{Files involved:}
\texttt{report/rubric\_checklist.md}, \texttt{PROJECT\_DECISIONS\_LOG.md}

\subsection{Step 1: Reproducible Project Skeleton}
\textbf{What was done:}
Created standardized directories for data, outputs, source modules, report assets, and app integration.
Added pipeline runner \texttt{run\_pipeline.py} and config-first architecture.

\textbf{Why:}
Team collaboration requires predictable file locations and consistent interfaces.

\textbf{Goal:}
Any teammate can run the same sequence and get the same artifacts.

\textbf{Files involved:}
\texttt{run\_pipeline.py}, \texttt{config/*.yaml}, \texttt{src/*.py}, \texttt{outputs/*}

\subsection{Step 2: Dataset Integration (Adult Census Income)}
\textbf{What was done:}
Integrated dataset download via OpenML using \texttt{scripts/download\_adult.py}; stored final file at \texttt{data/raw/adult.csv}.

\textbf{Why:}
Automated ingestion reduces setup errors across teammates and ensures consistent schema.

\textbf{Goal:}
One command should bootstrap the dataset for all team members.

\textbf{Important function path:}
\texttt{scripts/download\_adult.py} uses OpenML fetch and writes standardized column names.

\subsection{Step 3: Preprocessing Implementation}
\textbf{What was done:}
Implemented full preprocessing in \texttt{src/preprocessing.py}:
\begin{itemize}
  \item missing-token replacement (\texttt{?} and empty strings),
  \item whitespace normalization for categories,
  \item duplicate removal,
  \item target normalization to consistent labels,
  \item stratified train/val/test split,
  \item export of cleaned split files and a preprocessing summary table.
\end{itemize}

\textbf{Why:}
Modeling and statistics are unreliable without deterministic cleaning and split logic.

\textbf{Goal:}
Create stable, reusable, clean datasets for all downstream modules.

\textbf{Artifacts produced:}
\begin{itemize}
  \item \texttt{data/processed/train.csv}
  \item \texttt{data/processed/val.csv}
  \item \texttt{data/processed/test.csv}
  \item \texttt{outputs/tables/preprocessing\_summary.csv}
\end{itemize}

\subsection{Step 4: EDA (Exploratory Data Analysis)}
\textbf{What was done:}
Implemented \texttt{src/eda.py} for:
\begin{itemize}
  \item univariate statistics and visualizations,
  \item bivariate relations (correlation, numeric-target, categorical-target),
  \item multivariate PCA projections and loadings.
\end{itemize}

\textbf{Why:}
EDA is required by the rubric and essential to understand model behavior and privacy patterns.

\textbf{Goal:}
Extract interpretable behavioral signatures before moving into attack/defense.

\textbf{Main interpretation from EDA:}
\begin{itemize}
  \item \texttt{age}, \texttt{education\_num}, \texttt{hours\_per\_week} shift higher for \texttt{>50K},
  \item \texttt{capital\_gain}/\texttt{capital\_loss} are zero-inflated with rare strong outliers,
  \item numeric linear correlations are low, suggesting non-linear structures,
  \item PCA gives partial separation with overlap,
  \item class and demographics are imbalanced, relevant for fairness and threshold interpretation.
\end{itemize}

\subsection{Step 5: Reliability Stage}
\textbf{What was done:}
Implemented and executed \texttt{src/reliability.py}:
\begin{itemize}
  \item repeated stratified runs (\texttt{n\_runs=10}),
  \item logistic baseline with full preprocessing pipeline,
  \item per-run metrics + aggregate stats + 95\% confidence intervals,
  \item calibration curve export.
\end{itemize}

\textbf{Why:}
One-run results can be unstable. Reliability proves repeatability and scientific trustworthiness.

\textbf{Goal:}
Establish stable baseline metrics to justify next privacy attack conclusions.

\textbf{Reliability results summary:}
\begin{itemize}
  \item Accuracy: 0.8067
  \item Precision: 0.5643
  \item Recall: 0.8454
  \item F1: 0.6768
  \item ROC-AUC: 0.9050
  \item Brier: 0.1286
\end{itemize}
Confidence intervals were narrow, indicating strong run-to-run stability.

\subsection{Step 6: MIA (Membership Inference Attack) Baseline}
\textbf{What was done:}
Implemented and executed \texttt{src/mia\_shadow.py}:
\begin{itemize}
  \item trained one target model and multiple shadow models,
  \item extracted attack features from output probabilities,
  \item trained an attack model to predict member vs non-member,
  \item exported attack metrics, confusion matrix, feature importances, and MIA figures.
\end{itemize}

\textbf{Why:}
This is the core privacy objective: test if training membership is inferable from model behavior.

\textbf{Goal:}
Quantify privacy leakage risk with measurable attack performance.

\textbf{MIA result summary:}
\begin{itemize}
  \item Attack ROC-AUC: 0.6293
  \item Attack accuracy: 0.6110
  \item Attack precision: 0.5662
  \item Attack recall: 0.9487
  \item Attack F1: 0.7092
  \item Risk label (current rule): low
\end{itemize}

\textbf{Interpretation:}
Leakage signal exists (AUC $>$ 0.5) but remains below the moderate-risk threshold (0.65) in the current setup.

\section{What Each Important File Does}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.34\linewidth} >{\raggedright\arraybackslash}p{0.58\linewidth}}
\toprule
File & Role in the project \\
\midrule
\texttt{run\_pipeline.py} & Orchestrates step execution order and writes run summary JSON. \\
\texttt{config/dataset\_adult.yaml} & Central configuration for schema, split, reliability, MIA, and future defense settings. \\
\texttt{config/experiments.yaml} & Experiment-level settings and output directory contracts. \\
\texttt{scripts/download\_adult.py} & Fetches Adult dataset from OpenML and standardizes columns into local CSV. \\
\texttt{src/ingestion.py} & Loads raw CSV according to config and validates dataset availability. \\
\texttt{src/preprocessing.py} & Cleans data, normalizes target, performs stratified split, exports processed files. \\
\texttt{src/eda.py} & Generates descriptive statistics and exploratory figures/tables. \\
\texttt{src/reliability.py} & Runs repeated baseline experiments and computes stability/calibration evidence. \\
\texttt{src/mia\_shadow.py} & Implements shadow-model MIA and leakage-risk quantification. \\
\texttt{PROJECT\_DECISIONS\_LOG.md} & Append-only audit log of all decisions, changes, and rationale. \\
\texttt{report.md} & This living scientific report in LaTeX syntax, updated continuously. \\
\texttt{outputs/reports/pipeline\_summary.json} & Machine-readable summary of latest run and key metrics. \\
\bottomrule
\end{longtable}

\section{Libraries Used and Why They Matter}
\subsection{Core Data/Math}
\begin{itemize}
  \item \texttt{pandas}: tabular data loading, cleaning, export.
  \item \texttt{numpy}: vectorized numeric operations.
  \item \texttt{scipy}: statistical functions (confidence interval helper, tests in EDA/reliability workflows).
\end{itemize}

\subsection{Modeling and Attack}
\begin{itemize}
  \item \texttt{scikit-learn}: preprocessing pipelines, classical ML models, metrics, splitters, calibration and ROC tooling.
\end{itemize}

\subsection{Visualization}
\begin{itemize}
  \item \texttt{matplotlib} and \texttt{seaborn}: all exported figures for report and PPT.
\end{itemize}

\subsection{Configuration and Reproducibility}
\begin{itemize}
  \item \texttt{PyYAML}: config-driven architecture.
\end{itemize}

\section{Important Functions and Why They Exist}
\subsection{Preprocessing}
\begin{itemize}
  \item \texttt{\_normalize\_string\_columns}: trims string categories to avoid artificial category duplication.
  \item \texttt{\_normalize\_target}: enforces consistent target labels.
  \item \texttt{\_split\_dataset}: deterministic stratified splitting.
\end{itemize}

\subsection{EDA}
\begin{itemize}
  \item \texttt{\_plot\_numeric\_histograms}, \texttt{\_plot\_categorical\_bars}: univariate inspection.
  \item \texttt{\_plot\_correlation\_heatmap}, \texttt{\_plot\_boxplots\_by\_target}: bivariate structure.
  \item PCA block in \texttt{run}: multivariate structure and explained variance.
\end{itemize}

\subsection{Reliability}
\begin{itemize}
  \item \texttt{\_build\_model\_pipeline}: standardized baseline learning pipeline.
  \item \texttt{\_confidence\_interval}: quantifies uncertainty across repeated runs.
  \item \texttt{\_plot\_calibration}: checks probability quality.
\end{itemize}

\subsection{MIA}
\begin{itemize}
  \item \texttt{\_build\_attack\_features}: converts model-output behavior into attack features.
  \item \texttt{\_collect\_member\_features}: builds member/non-member training signals.
  \item \texttt{\_plot\_roc} and \texttt{\_risk\_level}: transforms raw attack outputs into interpretable privacy-risk evidence.
\end{itemize}

\section{Current Results and Interpretation in Plain Language}
\subsection{Preprocessing}
Data is clean, consistent, and split correctly.
Only 52 rows removed as duplicates from 48,842 total rows, so dataset integrity is preserved.

\subsection{EDA}
The dataset contains meaningful signal for the target class but not purely linear patterns.
This explains why flexible models and attack features based on confidence are useful.

\subsection{Reliability}
Baseline learning behavior is stable across repeated runs.
This is critical because unstable baselines invalidate later privacy conclusions.

\subsection{MIA}
There is measurable leakage signal, but not yet at a severe-risk level in current settings.
This is a valid and useful finding, because defense stage can now test whether risk can be further reduced or if utility shifts.

\section{Rubric Status (Up To This Point)}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.62\linewidth} >{\raggedright\arraybackslash}p{0.3\linewidth}}
\toprule
Criterion & Status \\
\midrule
Python script, importation, cleaning & Implemented \\
Univariate descriptive statistics & Implemented \\
Bivariate descriptive statistics & Implemented \\
Multivariate descriptive statistics & Implemented \\
Reliability analysis & Implemented with metrics and interpretation \\
Supervised/unsupervised baseline block & Implemented baseline and MIA pipeline \\
Innovative tests & MIA implemented; defense pending \\
Association rules & Pending \\
Clustering & Pending \\
Final communication package (PPT/report polish) & In progress \\
\bottomrule
\end{longtable}

\section{Team Handoff: What Teammates Should Do Next}
\subsection{Teammate Track A (Defense + Privacy-Utility)}
\begin{itemize}
  \item Implement \texttt{src/defense\_dp.py}
  \item Compare baseline vs defended model for:
  attack AUC, utility AUC/accuracy, and privacy-utility curve.
  \item Export tables and defense figures into \texttt{outputs/tables} and \texttt{outputs/figures}.
\end{itemize}

\subsection{Teammate Track B (Association Rules + Clustering)}
\begin{itemize}
  \item Implement \texttt{src/association\_rules.py} using Apriori.
  \item Implement \texttt{src/clustering.py} with K-Means and/or DBSCAN.
  \item Produce interpretable outputs tied to leakage behavior or target profiles.
\end{itemize}

\subsection{Teammate Track C (Dashboard + Final Communication)}
\begin{itemize}
  \item Expand \texttt{app/streamlit\_app.py} to load and display generated artifacts.
  \item Build final presentation narrative:
  problem $\rightarrow$ method $\rightarrow$ findings $\rightarrow$ limitations $\rightarrow$ recommendations.
  \item Maintain one explicit ``Rubric Coverage'' slide.
\end{itemize}

\section{What You Personally Did and Why (Owner Narrative)}
\begin{itemize}
  \item You established the reusable technical base first to avoid unstructured notebook drift.
  \item You implemented preprocessing because all later analyses depend on clean, consistent data.
  \item You implemented EDA to satisfy descriptive rubric requirements and build intuition for attack behavior.
  \item You implemented reliability to ensure your conclusions are stable and defendable scientifically.
  \item You implemented MIA baseline to answer the central privacy question of the project.
  \item You maintained an audit log and living report to preserve traceability and team continuity.
\end{itemize}

\section{What Is Still Needed to Reach the End Goal}
\begin{enumerate}
  \item Defense experiments and privacy-utility tradeoff finalization.
  \item Association-rule mining with interpretation.
  \item Clustering block with visual interpretation.
  \item Streamlit dashboard integration for live demo.
  \item Final report polishing and PPT final narrative.
\end{enumerate}

\section{Definition of Done}
This project is considered complete when:
\begin{itemize}
  \item all rubric technical blocks are implemented and documented,
  \item all major outputs are reproducible via pipeline commands,
  \item report and PPT clearly explain methods and findings,
  \item privacy leakage and defense conclusions are evidence-based and traceable.
\end{itemize}

\section{Artifacts Reference (Current)}
\begin{itemize}
  \item Run summary: \texttt{outputs/reports/pipeline\_summary.json}
  \item Reliability tables: \texttt{outputs/tables/reliability\_*.csv}
  \item Reliability figures: \texttt{outputs/figures/reliability\_*.png}
  \item MIA tables: \texttt{outputs/tables/mia\_*.csv}
  \item MIA figures: \texttt{outputs/figures/mia\_*.png}
  \item EDA tables: \texttt{outputs/tables/eda\_*.csv}
  \item EDA figures: \texttt{outputs/figures/eda\_*.png}
\end{itemize}

\end{document}
