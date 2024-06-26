<div align="right">

[Back to Table of Contents](README.md#Table-of-Contents)

</div>

# Python Calculation Jupyter Notebooks
This supplemental material includes a series of Jupyter Notebooks. These are itemized below.

<h3>Z01.1: Preparation of Study Corpus</h3>
This notebook works through the preparation of a clean study corpus by 
 extraction of data from the transliterated the Voynich Manuscript.

[Go to notebook](./Z01.1_Preparation_of_Study_Corpus.ipynb)


<h3>Z01.2: Token Cohorts</h3>
This notebook sets up the several cohorts of tokens used for the study.

[Go to notebook](./Z01.2_Token_Cohorts.ipynb)
  
<h3>Z01.3: Token Length Analysis</h3>
This notebook shows the calculations for the analysis of token length distributions for the different cohorts.

[Go to notebook](./Z01.3_Token_Length_Analysis.ipynb)


<h3>Z01.4: Token Propensities Analysis</h3>
This notebook shows the calculations for the analysis of token usage propensities for each of the subject cohorts.

[Go to notebook](./Z01.4_Token_Propensities_Analysis.ipynb)
  
<h3>Z01.5: Extra Analyses</h3>
this notebook contains additional calculations and charts not directly discussed in the paper.
    
[Go to notebook](./Z01.5_Extra_Analyses.ipynb)

<h2>Python Dependencies</h2>
Python 3.10 has been used for all analyses.
All of the Python library modules used in the notebooks are generally available, except for two: <code>voynichlib</code> and <code>qlynx</code>.

<h3><code>qlynx</code></h3>
The <code>qlynx</code> is a proprietary library providing some general utility and plotting functions. These 
that can be easily reproduced or replaced by
other routines and none of the analysis is dependent on its use.

<h3><code>voynichlib</code></h3>
The <code>voynichlib</code> is a comprehrensive library
developed for the purposes of managing and parsing  Voynich Manuscript transliteration files and for processing
the elemental constructs of the Voynichese data.  This module is for internal use at this time, but is
being prepared for potential availability to the Voynich Manuscript research community.




<div align="right">

[Back to Table of Contents](README.md#Table-of-Contents)

</div>