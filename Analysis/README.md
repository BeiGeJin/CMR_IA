## ANALYSIS FILE SUMMARY

### Part 1
- simu1: item recognition, recency and similarity effect (Halpern et al., 2022)
- simu2: item recognition, contiguity effect on successive probes (Schwartz et al., 2005)
- simu3: item & association recognition, difference in forgetting rate
- simu4: item recognition, word frequency effect

### Part 2
- simu5: cued recall, recency effect (Murdock, 1967)
- simu6: cued recall, associative symmetry (Kahana, 1993; Kahana, 2002)
- simu7: cued recall, PLI and ILI (Davis et al., 2008)
- simu8: cued recall, similarity effect and ILI (Pantelis et al., 2008)

### Part 3
- simuS1: successive tests on recognition and cued recall
- simuS2: successive tests on recognitions

## Insights

### Part 1
- $c$ is working memory. $M^{FC}$ is long-term memory.
- For recognition, parameters in effect are typically only five: $\beta_{enc}$, $\beta_{cue}$, $\gamma_{fc}$, $s_{fc}$, $c_{thresh}$. $\beta_{rec}$ is used when retrieval-while-learning is True, but it is typically False. It is deterministic. It does not have a mechanism for primacy effect.
- $\beta$ controls how much $c$ is updated when an item is coming. If $\beta = 1$, every new item overhauls c. If $\beta = 0$, every new item does not change $c$. In either situation, there is no recency effect, and $c$ would not help memory. The influence of $\beta$ on memory is a curve, the middle is optimal. But where the peak is depends on list length.
- There is also a balance in $\gamma$. The diagnoal element in $M$ is $1-\gamma$, which means that a higher $\gamma$ results in a lower direct input in $c$. Before encoding, this means a higher proportion of semantic. After encoding, this means a higher proportion of semantic + related context. But if $\gamma = 0$, there is no long-term studying.
- In *simu1*, $\beta_{enc}$ and $\beta_{rec}^{post}$ is set to be quite low, to make sure a resonable performance after a long lag. $\beta_{cue}$ is set to be 0, and actually has no effect whatever its value is, because during recognition we only need $c_{in}$ and compare $c_{in}$ with $c_{old}$, so $\beta_{cue}$ is not involved.
- In *simu2*, to get a resonable distribution of context similarity (new items are not too low), also because the semantic matrix is randomly generated in the range of [0, 0.1], we set a high $s_{fc}$ and a low $\gamma_{fc}$. $\beta_{cue}$ has an effect on the irregular tail of the ROC curve.
- In *simu3*, $d_{ass}$ and $c_{thresh}^{ass}$ are new parameters for association recognition. $s_{fc}$ is set to be a bit high to obtain a resonable distribution of context similarity (new items are not too low). Other parameters are quite normal.
- In *simu4*, a few new mechanisms are added. Initially, as high frequency words are correlated with low average semantic associations, HF items tend to have lower FAR. Thus, a criterion-shift mechanism is added to lower the threshold for HF words and obtain a positive correlation between FAR and word frequency. Elevated-attention mechanism is then added to heighten the HR of LF words.

### Part 2
- Coming to cued recall, the full set of parameters comes into play, e.g., those for the leaky accumulator and primacy effect. Now there is also retrieval variability because of the random process in the leaky accumulator.
- To assess the outcomes, $f_{dif}$ (i.e., $f_{in}$ - ret_thresh) is the most reliable indices. Better than only $f_{in}$, which might be confused by the current cue and the previous item.
- nitems_in_accumulator is also an important parameter that is easily neglected (in Polyn et al. 2015, it is 4 times list length). It interacts with $\lambda$, the lateral inhibition parameter. A large amount of items with a fixed $\lambda$ means a lot of No-recalls. Changing the $\kappa$ also has an effect on No-recalls. $\eta$ is where randomness comes from. Higher $\eta$ results in less correct. I typically set $\eta = 0$ to test the basic trend at the beginning.
- In *simu5*, $\phi_s$ controls the overall $f_{dif}$ and correct rate while $\phi_d$ controls the curve of last three. A higher $\phi_d$ results in an upper tail, mediated by $\phi_s$. I forget why $\gamma$ should be inbalanced and high.
- In *simu6a*, I try to keep the study parameters the same as *simu5*, except the primacy parameters that are important for the curve of last three. I once encountered the problem that the decreasing at the beginning was not straight enough. I now make a compromise by making the correct rate at lag 0 to be 0.8 rather than closed to 1.0. 
- In *simu6b*, things are mystery. I unexpectedly set $c_{thresh}$ to be high (screening intrusions) and get the correct output. Without this, there would be a higher Q for Incongruent than Congruent, and an often higher inconsistency between Test1 and Test2. The screening mechanism is to get those inconsistent answers to be consistenly false. Also I deviate from the initial paper that all items are repeated once.
- In *simu7*, $\gamma_{fc}$ and $\gamma_{cf}$ turns out to be the most important parameters for ILI. Attempts (mysteriously) reveals that $\gamma_{cf}$ - $\gamma_{fc} = 0.2$ could produce the foward asymmetry. Higher the difference, higher the foward asymmetry. Primacy parameters may influence this difference. For PLI, $\alpha$ is important, which determines when an item could be retrieved again. If $\alpha = 0.5$, the curve would be flat. If $\alpha = 0.9$, the curve will peak at arond list 5, because that is where the retrieval threshold restores. $\alpha = 1$ seems to best produce the downwarding PLI curve. nitems_in_accumulator also has an effect, if too low, list 1 would boost.
- In *simu8*, nitems_in_accumulator must be 16 (all items in one session) and No_recall should be set to be np.arange(0,8) to prevent recalling faces (the task is to recall names given faces). $\beta_{cue}$ and $d_{ass}$ seems to be important for overall accuracy. 

### Part 3