
<div align="right">

[Back to Table of Contents](README.md#Table-of-Contents)

</div>

# Response to Reviewers' Comments

Anonymous reviewer comments provided very helpful 
and much appreciated feedback.

Some of this feedback was incorporated when making  final 
revisions to the 
paper, but not all of it. In most cases where we did not make 
a change to the paper's
content, it was because the comment was not suggesting any, or 
it was just not feasible due to the 10-page limitation imposed 
for publication. 
But in a few other cases, where we felt the comment's concern 
was not actually applicable, we felt it still deserved more than 
the appearance of 
being ignored. We have therefore provided some additional response to those below.

------

<ol>
<li><b>Comment:</b> "inspired guesswork" - Linear B is a good example of this;
a combination of painstaking work by Kober and Ventris and some lucky guesses<br>
<br>
<p>
<b>Response:</b> 
<br>
<ul>
It is indeed a good example. As a matter of fact, 
the references to inspired guesswork in the paper 
were partly inspired by Ventris's work, and in particular by
a quote of his collaborator, John Chadwick:
<ul>"The achievement of decipherment … required 
painstaking analysis and sound judgment, but at the same 
time an element of genius, the ability to take a leap in the 
dark, but then to find firm ground on the other side. Few 
discoveries are made solely by processes of logical deduction. 
At some point, the researcher is obliged to chance a guess, 
to venture an unlikely hypothesis; what matters is whether he 
can control the leap of the imagination, and have the honesty 
to evaluate the results soberly. Only after the leap is made 
is it possible to go back over the working and discover the 
logical basis which provided the necessary springboard."<br>
 <div align="right">pg 14 of "The Man Who Deciphered Linear B" by Andrew Robinson</div>
</ul>
</ul>

------

<li><b>Comment:</b> we have no idea about the gender of the scribes; 
please use a gender-neutral term
<p>
<br>
<b>Response:</b> 
<br>
<ul>
We kept the use of the pronoun "he" because doing otherwise would provide a cognitive distraction while 
serving no apparent purpose relevant to the study. 
This would not be the case if there was some reason to believe that the Voynich scribe might be female, or if 
the possibility of it played
some role in the study and its analyses.  Neither is the case here.
The vast majority of scribes in the medieval period  (and for a long time after) 
were male. In fact, although possible, it would be extraordinary if a Voynich scribe turns out otherwise. 
And in traditional English, a masculine pronoun is used not only
when the referent is known to be male, but also when the gender is unknown or 
contextually irrelevant, while a female pronoun implies some significant expectation that the 
referent may be specifically female.
</ul>

------

<li><b>Comment:</b> Was there a correction in p values for multiple tests? 
(e.g. Bonferroni correction or some other post-hoc test?) I wondered 
also why Chi-sq tests and not, say, mixed effect modeling, are the 
appropriate test to use here.
<p>
<br>
<b>Response:</b> 
<p>
<ul><li>A Bonferroni correction is used to adjust the p-value threshold in order 
to make statements about the results of a whole set of hypothesis 
tests while still basing those statements on a single unadjusted 
p-value threshold. It is applicable when multiple tests are being 
applied to multiple datasets (or the same datasets multiple times) 
to test for  the SAME hypothesis. This fact is sometimes sloppily 
described, however, resulting in the misconception that it should be 
used when multiple different hypotheses are being tested. 
(Wikipedia is one example of this, since it describes the need 
for a correction using the 
phrase “If multiple hypotheses are tested…” implying that the hypotheses 
themselves are different.) 
<br>
Such a correction does not apply in our case because each test is 
being applied to a different data set to assess an independent 
hypothesis about each one. For example:
<ol>
<li>What is the probability that the differences between cohorts A and C are due to random sampling?  
and 
<li>What is the probability that the differences between cohorts B and C are due to random sampling?
</ol><br>
These are two independent hypotheses (and, in addition, our 
data cohorts are specifically mutually 
independent.) If the tests in case (a) suggest a 
probability (p-value) of 0.1, 
it has no bearing 
on the probability involved in case (b). 
(And we would not expect subsequent tessts on additional cohorts
to retroactively affect earlier ones.)
<br>
The <em>form</em> of the hypothesis in each of our tests is the same, but 
the hypotheses themselves are different. This is similar to lie detector
tests.
A lie detector always tests the same 
form of a hypothesis – that the test subject is lying – 
but each test on a different subject
is a different hypothesis; the results of a 
test on one individual will have no impact on that of another individual.  
Only if you are specifically asking whether BOTH subjects are lying do the 
probabilities combine with the need to adjust the p-value threshold being used.
<br><br>
(It is also worth keeping in mind that strictly speaking, 
a p-value test is NOT answering “What is the probability that 
the null hypothesis is true?”, but rather “What is the probability 
that the differences are due to chance sampling?”)
<p>

<br>
<li>Mixed effects modeling is for cases involving 
multiple  input variables (in this context, causes) to predict an 
output (in this context, a statistical result). We have not seen how it 
would or should be used in our case.
<p>
<br>
<li>A Chi-Squared (aka “Chi-square”) test is used to test if there's a 
significant difference between what we expect to see and what is actually observed for a particular situation or phenomenon. This is directly applicable to our case.





<div align="right">

[Back to Table of Contents](README.md#Table-of-Contents)

</div>