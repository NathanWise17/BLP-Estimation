# BLP-Estimation

I finished the project using both R and Python. The general workflow, in terms of files, was as follows: 
\begin{enumerate}
    \item maincode.rmd
    \begin{enumerate}
        \item Performed operations on the main dataset, such as generating marketshares
        \item Performed the Logit and Nested Logit estimation
    \end{enumerate}
    \item lags.py
    \begin{enumerate}
        \item Generated the lagged wholesale price instrument
        \item Readied the data for BLP Estimation
    \end{enumerate}
    \item BLP Estimation.py
    \begin{enumerate}
        \item Performed the BLP Estimation, returning estimates for $\alpha$,$\pi$,$\sigma$
    \end{enumerate}
    \item elasticitycode.py
    \begin{enumerate}
        \item Generated the elasticities for the Logit and Nested Logit models
    \end{enumerate}
    \item BLPElasticities.py
    \begin{enumerate}
        \item Generated the elasticites for the Random Coefficients Model
    \end{enumerate}
    \item Markups.py
    \begin{enumerate}
        \item Calculated the markups given the elasticity estimates
    \end{enumerate}
    \item Elasticites.rmd
    \begin{enumerate}
        \item Generated the graphs for all elasticites
    \end{enumerate}
    \item Markups.rmd
    \begin{enumerate}
        \item Generated the graphs for the markups
    \end{enumerate}
\end{enumerate}
