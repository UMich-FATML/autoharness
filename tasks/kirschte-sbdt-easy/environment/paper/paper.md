# S-BDT: Distributed Differentially Private Boosted Decision Trees 

Thorsten Peinemann[∗1] , Moritz Kirschte[∗1] , Joshua Stock[2] , Carlos Cotrini[3] , and Esfandiar Mohammadi[1] 

1 Universität zu Lübeck, Lübeck, Germany 

> 2Universität Hamburg, Hamburg, Germany 

> 3ETH Zurich, Zurich, Switzerland 

> 1 _{t.peinemann, m.kirschte, esfandiar.mohammadi}@uni-luebeck.de_ 

> 2 _joshua.stock@uni-hamburg.de_ 

> 3 _ccarlos@inf.ethz.ch_ 

## **Abstract** 

We introduce S-BDT: a novel ( _ε, δ_ )-differentially private distributed gradient boosted decision tree (GBDT) learner that improves the protection of single training data points (privacy) while achieving meaningful learning goals, such as accuracy or regression error (utility). S-BDT uses less noise by relying on non-spherical multivariate Gaussian noise, for which we show tight subsampling bounds for privacy amplification and incorporate that into a Rényi filter for individual privacy accounting. We experimentally reach the same utility while saving 50% in terms of epsilon for _ε ≤_ 0 _._ 5 on the Abalone regression dataset (dataset size _≈_ 4 _K_ ), saving 30% in terms of epsilon for _ε ≤_ 0 _._ 08 for the Adult classification dataset (dataset size _≈_ 50 _K_ ), and saving 30% in terms of epsilon for _ε ≤_ 0 _._ 03 for the Spambase classification dataset (dataset size _≈_ 5 _K_ ). Moreover, we show that for situations where a GBDT is learning a stream of data that originates from different subpopulations (nonIID), S-BDT improves the saving of epsilon even further. 

## **1 Introduction** 

We present a differentially private distributed learning algorithm for a class of fast learners, the so-called gradient boosted decision tree ensembles (GBDT): these models combine many weak decision tree learners into an ensemble to prevent overfitting while still capturing complex nonlinear patterns. GBDTs utilize a robust, data-efficient, incremental learning method, traits that are well-suited for strong privacy-preserving approximations. 

Classical decision trees are vulnerable to privacy attacks on training data [16]. Each tree consists of splits and leaves which are data-dependent and typically significantly overfitted and thus potentially leak information. The state-of-the-art notion for provably protecting against such information leakage is ( _ε, δ_ )-differential privacy (DP), which requires that the impact of single data points be small and deniable. To protect the training itself, user data should be stored locally and not collected by a trusted 3rd party. This can be realized using distributed training. Yet, training GBDTs in a privacy-preserving manner needs to achieve strong utility-privacy tradeoffs: protect single training data points (privacy) while keeping the strong machine learning performance (e.g., high classification ac- 

> ∗The first two authors equally contributed to this work. 

curacy or low regression error) of GBDTs (utility). 

The best-performing prior work [24] has shown that training GBDTs has the potential to achieve this strong utility-privacy tradeoff. Yet, we are able to achieve better privacy guarantees (lower _ε_ ) for the same utility for regular training as well as in a setting where the data distribution changes during training, i.e. a stream of data that originate from different subpopulations (non-IID): For such a stream, it is crucial to keep the information from data that arrived at previous rounds, but prior work cannot use that information to avoid additional privacy costs. 

Lower _ε_ values for the same degree of utility are desirable, as for many DP algorithms _ε_ is in _O_ ([1] _/n_ ), for _n_ many data points. Hence, achieving such a better privacy-utility tradeoff means that such a DP algorithm can be applied to smaller datasets. Thus, improving _ε_ without impeding utility is of practical importance. 

**Contributions** We introduce the novel securely distributed DP gradient boosted decision tree algorithm S-BDT[1] that achieves stronger utility-privacy tradeoffs than the best-performing prior work [24] for strict privacy requirements ( _ε ≤_ 0 _._ 5) and regression tasks, and provides significant improvements for a stream of non-IID data. In contrast to prior work, S-BDT incorporates three techniques that improve privacy: (1) subsampling to increase the number of trees, (2) leaf-balanced noise for better noise calibration, and (3) Rényi filters for individual privacy accounting. 

- (1) _Novel condition for exact Rényi DP Bounds for Poisson Subsampling._ Subsampling is known to improve DP bounds. For simple DP algorithms, prior work has shown tight bounds for privacy loss, which leads to less noise and hence better utility. Prior work only showed untight bounds for more general DP algorithms such as used in this work (details in Sec. 6.3). We introduce a novel condition that enables us to prove tight privacy bounds for subsampling for our algorithm. This novel condition generalizes prior results and might be of independent interest. 

- (2) _Individual Rényi DP Bounds for Leaf-balanced Noise._ We show individual RDP for a multivariate Gaussian that balances the scale of the noise added to the first derivative and second derivative of the loss that Newton Boosting requires to construct a single leaf. These 

   - 1code available at https://github.com/kirschte/sbdt 

1 

**==> picture [486 x 162] intentionally omitted <==**

**----- Start of picture text -----**<br>
Ensemble after 1 Ensemble after Ensemble after Ensemble after  ID Privacy BudgetClassical Ensemble after Tree  700 + 200  (test AUC:  81% ):<br>Tree  0 : DP Initial Score Tree  1 : Tree  10 : Tree  700 : -1--2-… 100%100%… -1--2-ID Privacy BudgetClassical 100% 100% [In Subsample?] ✘✘<br>… … without  Rényi Filter … … …<br>-A- 100% ✘<br>-B- 100% ✘<br>2 ID Privacy BudgetIndividual … … …<br>Leaf-balanced noise -1--2- 50%90% Ensemble after Tree  700 + 800  (test AUC:  99% ):<br>ID Privacy BudgetIndividual [In Subsample?] 3 ID Privacy BudgetIndividual [In Subsample?] ID Privacy BudgetIndividual [In Subsample?] … … ID Privacy BudgetIndividual [In Subsample?]<br>-1- 2% ✘ … -1- 10% ✘ … -1- 50% with  Rényi Filter 4 -1- 75%<br>Class probability -2- 0% -2- 30% -2- 90% -2- 100% ✘<br>Data points … … … … … … … … … … … …<br>(2 classes) -A- 70% ✘<br>width Random split ∝ [1] /tree depth initialdata newnon-IID data5 -B-… 35%… …<br>DP leaf value -1-ID Privacy BudgetIndividual0% -A-ID Privacy BudgetIndividual0%<br>Legend -2-… 0%… -B-… 0%…<br>**----- End of picture text -----**<br>


Figure 1: **Schematic overview of S-BDT (** _ε_ = 0 _._ 5 **)** classifying two-dimensional concentric circles where the inner yellow circle arrives after the (here: 700) regular training rounds. The numbers ➀ to ➄ refer to S-BDT’s key technical features (cf. Sec. 2). 

individual RDP bounds neatly fit with subsampling via our contribution above. 

- (3) _Stream of non-IID data._ We adopt a so-called Rényi filter for individual privacy accounting to ensure that underutilized data points that arrived in the past can still be used for training the model, upon the arrival of novel data. We show for a stream of non-IID data that S-BDT can significantly improve on a naïve baseline which adds extra training rounds but discards formerly known data. 

- (4) _Strong empirical performance._ We experimentally show that our DP GBDT algorithm S-BDT provides a saving of _>_ 50% in terms of epsilon for _ε ≤_ 0 _._ 5 (Fig. 3) on Abalone regression (dataset size _≈_ 4 _K_ ) of _>_ 30% for _ε ≤_ 0 _._ 08 on Adult classification (dataset size _≈_ 50 _K_ ), and of _>_ 30% for _ε ≤_ 0 _._ 03 on Spambase classification (dataset size _≈_ 5 _K_ ). Our ablation study evaluates the effectiveness of our improvements (1) to (3). 

## **2 Overview** 

We briefly sketch gradient boosted decision tree ensembles (GBDT) and then highlight our key technical contributions. As depicted in Fig. 1, decision trees store at each inner node a so-called _split_ that partitions the input space along their features. Each path from the root to a leaf thus describes one partition of the input space. For each of these partitions, a decision tree learns a constant leaf value. Upon prediction, features of the input are used to choose the corresponding leaf and predict its leaf value. GBDTs iteratively learn a sequence of trees, which leads to overlapping partitions of the input space. Upon prediction, a GBDT sums the leaf values of all trees. During boosted training, each new tree assigns each data point a new label which corrects the current ensemble prediction. 

The main sources of privacy leakage for GBDTs are choosing the best splits and leaves. S-BDT randomly chooses splits, as prior work [7, 24] has shown that this strategy in combination with the iterative improvement of boosting leads to acceptable results. 

Our key technical contribution is that we first combine _subsampling_ – a privacy amplification – with an improved 

noise calibration ( _leaf-balanced noising_ ) and individual privacy accounting (via _Rényi filters_ ) and second, derive tight privacy bounds for this combination which might be of independent interest. Our evaluation shows that our modifications lead to DP GBDTs with significantly better utility. 

➀ **DP Initial score.** We reduce the range of the leaf-value-corrections by first DP-approximating the labelmeans. 

➁ **Tight RDP bounds for leaf-balanced noise.** DP Newton Boosting for GBDTs [24] sets the leaf value as the fraction of the noisy first derivative of the training loss divided by the noisy second derivative. We propose _leaf-balanced noise_ that boosts the utility by balancing the magnitude of both noises. On the theoretical side, this noisy derivative pair represents a non-spherical multivariate Gaussian mechanism. We prove tight and individual Rényi DP bounds for this mechanism in Thm. 16, something prior work [10, 24] did not provide. For instance, MVG [10] uses untight bounds for a more general noise distribution: matrix-variate Gaussian. 

➂ **Tight RDP bounds for subsampled multivariate noising.** Choosing a random subset of data points (subsampling) is a known privacy amplification. Yet, prior subsampling results do not apply to DP GBDTs as they either offer untight generic bounds or require univariate noise. The untight bounds do not lead to any privacy amplification (e.g. a factor 15 worse than the tight bound for one of our runs). For a tight bound, Zhu and Wang [34] prove their tight subsampling bound only for univariate Gaussian noise, yet a DP GBDT adds 2-dimensional noise per leaf. Hence, we prove in Lem. 15 a general version for multivariate Gaussian noise including our leaf-balanced noise. 

➃ **S-BDT boosted by Rényi filters.** Feldman and Zrnic [14] have shown that in some cases data points whose information has not been used during learning can be re-used in later trees without additional privacy leakage. To this end, a so-called individual privacy accounting via Rényi filters is used. Fig. 2 illustrates how this technique utilizes the information of more data points. We prove that the tight Rényi bounds from above constitute a sound individual Privacy accountant (cf. Sec. 6.5) and incorporate a Privacy accountant into S-BDT (cf. Sec. 6.1). 

➄ **S-BDT is ready for a stream of non-IID data** 

2 

**==> picture [231 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 . 0<br>Individual<br>Rényi Filter<br>0 . 8 False<br>True<br>0 . 6<br>0 . 4<br>(average)<br>0 . 2<br>0 . 0<br>0 1000 2000 3000<br>data point index (sorted)<br>ε<br>budget<br>privacy<br>**----- End of picture text -----**<br>


Figure 2: **Individual Rényi filter (IRF) boosts the average privacy leakage** (dotted line) closer to the worstcase accounted one. The noise scale is calibrated on the regular tree training rounds, but data points that did not consume their accounted budget are used for free in extra rounds (here: 100 rounds). On the abalone dataset, we 1) measure for every data point (x-axis) the privacy leakage by how much the gradient sum of S-BDT’s leaf changes after removing one data point and 2) aggregate it across the ensemble (y-axis). 

**via a Rényi filter.** We pose the important challenge of consecutively updating a model with later arriving data which are useful to reflect recent data distribution shifts (non-IID). Regularly, a privacy budget once spent cannot be redeemed, thus classical techniques fail to reuse old data to update such a model. Yet, old data is needed to learn new decision boundaries between the old data and the new data points and to ensure that the old data is not unlearned. As a solution, we propose our S-BDT tailored with Rényi filters that keeps using those old data that did not consume their accounted budget together with later arriving data points at no additional privacy cost (cf. Sec. 8.4). 

➅ **Scalable distributed learning.** Our differentially private training algorithm for GBDT works seamlessly with distributed learning (cf. Sec. 6.7), a setting where multiple parties jointly train a model, without divulging their respective private training datasets to any other user or the public. This is especially relevant in the medical field with sensitive medical metadata that hospitals might want to keep on-premise but use to train a machine learning model collaboratively with other hospitals and their share of sensitive data. Our extension to distributed learning builds on prior work [24] that established distributed training of DP GBDT. 

## **3 Preliminaries** 

## **3.1 Differential Privacy** 

Differential privacy (DP) [12] is the de facto standard for provable privacy. Given a mechanism that has a datadependent output, DP requires that the impact of single data points in the output is limited and thus deniable. In line with close prior work [23, 24], we consider unbounded DP, where the effect of adding or removing a single instance from a dataset on the output is analyzed. 

**Definition 1** (Neighboring datasets) **.** Given two datasets 

_X ⊆X_ and _X[′] ⊆X_ where _X[′]_ := _X ∪{x}_ for some _x ∈X_ , then _X_ and _X[′]_ are neighboring: _X ∼x X[′]_ , or in short _X ∼ X[′]_ . 

**Definition 2** (DP, [12]) **.** A randomized mechanism _M_ : _X �→R_ satisfies ( _ε, δ_ )-DP if, for any two neighboring datasets _X ∼ X[′]_ and any observation _o_ : Pr[ _M_ ( _X_ ) = _o_ ] _≤ e[ε] ·_ Pr[ _M_ ( _X[′]_ ) = _o_ ] + _δ_ . 

For accounting, we use a variant of DP: ( _α, ρ_ ( _α_ ))-Rényi differential privacy (RDP) which bounds the _α_ -th moment of the privacy loss. With RDP we can mathematically describe features like composition (cf. Thm. 5), filter (cf. Sec. 3.3), or subsampling (cf. Sec. 3.4). Using Cor. 6, we can convert Rényi DP to DP. For a tight conversion, we refer to [31, Theorem 2] which requires access to all _α_ . 

**Definition 3** (Rényi Divergence, [29]) **.** Given two probability distributions _P, Q_ over _R_ where _P_ ( _o_ ) denotes the density of _P_ at _o_ , then the Rényi divergence of order _α_ is defined as 

**==> picture [150 x 15] intentionally omitted <==**

**Definition 4** (Rényi DP, Definition 4 in [26]) **.** A randomized mechanism _M_ : _X �→R_ satisfies ( _α, ρ_ ( _α_ ))Rényi DP if, for any two neighboring datasets _X ∼ X[′]_ : _Dα_ ( _M_ ( _X_ ) _||M_ ( _X[′]_ ) _≤ ρ_ ( _α_ ). 

**Theorem 5** (Adaptive sequential composition for RDP, [14] Theorem 3.1) **.** _Let α and ρ_ ( _α_ ) _be fixed, X be a dataset space, and M be a sequence of adaptively chosen mechanisms Mi_ : Π _[i] j[−]_ =1[1] _[R][j][× X][�→R][i][for][i][ ∈{]_[ 1] _[, . . . , k][ }][,][i.e.][M][i] has the outputs of all previous mechanisms R_ 1 _, . . . , Ri−_ 1 _as an input. If each Mi satisfies_ ( _α, ρi_ ( _α_ )) _-Rényi DP and_ � _i[ρ][i]_[(] _[α]_[)] _[ ≤][ρ]_[(] _[α]_[)] _[then][M][satisfies]_[(] _[α, ρ]_[(] _[α]_[))] _[-Rényi][DP.]_ 

Thm. 5 also holds if the mechanisms are not adaptively chosen, i.e. have privacy budgets fixed in advance. 

**Corollary 6** (RDP to DP, Thm. 21 in [4] or Prop. 12 in [9]) **.** _For any δ ∈_ [0 _,_ 1] _, if a mechanism is_ ( _α, ρ_ ( _α_ )) _-RDP, then it is_ ( _ε, δ_ ) _-DP with ε_ = _ρ_ ( _α_ )+ log( _[α] α[−]_[1][)] _[−]_[lo][g(] _[δ] α_[)+] _−_[lo] 1[g(] _[α]_[)] _._ 

If the output of a function _q_ , e.g. the leaf value in a GBDT, is _s_ -sensitivity bounded, then adding Gaussian noise to _q_ is RDP. 

**Definition 7** ( _L_ 2 sensitivity) **.** For neighboring inputs _X, X[′] ∈X_ , universe _U_ , and a randomized function _q_ : _X �→U_ , the _L_ 2 sensitivity _s_ is defined as _s_ := max _X∼X′ ∥q_ ( _X_ ) _− q_ ( _X[′]_ ) _∥_ 2. We call a function with a finite _L_ 2 sensitivity _s_ , an _s-L_ 2 _-sensitivity bounded function_ . 

**Theorem 8** (Gaussian mechanism, [13][26]) **.** _Let q_ : _X �→_ R _[m] be an s-L_ 2 _-sensitivity bounded function with respect to inputs X ∼ X[′] . The Gaussian mechanism M_ : _X �→_ R _[m] of the form M_ ( _x_ ) = _q_ ( _x_ ) + _N_ (0 _, s_[2] _σ_[2] _Im_ ) _satisfies_ ( _α,[α] /_ 2 _σ_[2] ) _- Rényi DP._ 

**Theorem 9** (Post-Processing [26]) **.** _Let M_ : _X �→R be_ ( _α, ρ_ ( _α_ )) _-RDP mechanism and f_ : _R �→R[′] a randomized function. Then, for any pair of neighboring datasets X ∼ X[′]_ 

**==> picture [198 x 11] intentionally omitted <==**

_i.e. Rényi differential privacy is preserved by postprocessing._ 

3 

|Symbol|Description|
|---|---|
|_M_|Randomized mechanism|
|(_x, y_)_∈X_ (or _D_)|labeled data point of dataset|
|_X ∼X′_, _X ∼x X′_|Neighboring datasets (difering in _x_)|
|_Dα_(_M_(_X_)_||M_(_X′_))|Rényi Divergence of order _α_|
|(_α, ρ_(_α_))|Rényi DP bound _ρ_(_α_) for _α_|
|(_α, ρ_<br>(_i_)<br>_t_ (_α_))|Individual RDP bound of _xi_ for|
||round _t_|
|_Fα,ρ_(_α_)|Individual Rényi flter|
|˜_yi_|Prediction for _xi_|
|(_gi, hi_)|Gradient and Hessian of _xi_|
|_g∗, h∗, m∗_|Gradient/Hessian/label clipping|
||bound|
|_r_1_, r_2|Noise weights for leaf value|
|_γ_|Subsampling ratio|
|_η_|Learning rate|
|_λ, β_|Regularization parameters|
|_ε_trees_, ε_init_, ε_ds|_ε_ privacy budget shares|
||for trees/initial score/dataset size|
|_σ_2<br>leaf|Unweighted variance of leaf Gaussian|
|_T_regular_, T_extra|Number of regular/extra training|
||rounds|



Table 1: **Notation Table** 

## **3.2 Gradient Boosted Decision Trees** 

GBDT [17] learns a sequence of decision trees by iteratively correcting errors of prior trees. Let _X_ = _{_ ( _x_ 1 _, y_ 1) _, . . . ,_ ( _xn, yn_ ) _} ⊆_ R _[m] ×_ R denote a labeled dataset with _n_ data points and _m_ features. For simplicity, we denote _y_ as a label although a regression target applies equally. A tree ensemble model _ϕk_ := � _f_ 1 _, . . . , fk_ � minimizes 

**==> picture [199 x 13] intentionally omitted <==**

where _l_ is a twice-differentiable convex loss function, e.g. squared error or binary cross-entropy, that measures the difference between the prediction _y_ ˜ _i_ := _ϕk_ ( _xi_ ) and the label _yi_ , and Ω( _ft_ ) =[1] _/_ 2 _λ∥Vt∥_[2] is a regularization term on the leaves vector _Vt_ = Leaves( _ft_ ). 

Ensemble _ϕk_ uses _k_ trees ( _ft_ ) _t[k]_ =1[to][predict] _[y]_[˜] _i_[for] _[x] i_[:] 

**==> picture [175 x 12] intentionally omitted <==**

where _η_ is a learning rate and _β_ a regularization parameter for the clipping routine clamp( _z, −β, β_ ) = max( _−β,_ min( _z, β_ )). 

For training each tree _ft_ , XGBoost [11] proposes Newton boosting, a second-order approximation of the loss function: 

**==> picture [161 x 46] intentionally omitted <==**

with gradient _gt_ ( _y_ ˜ _i, yi_ ) = _∂/∂y_ ˜ _il_ ( _y_ ˜ _i, yi_ ) and Hessian _ht_ ( _y_ ˜ _i, yi_ ) = _∂_[2] _/∂y_ ˜ _i_[2] _[l]_[(] _[y]_[˜] _[i][, y][i]_[)][.] If we have a squared error loss function like for regression, we have a closed ˜ ˜ form of the gradient _gt_ ( _yi, yi_ ) = _yi − yi_ and Hessian _ht_ ( _y_ ˜ _i, yi_ ) = 1. If we have a binary cross-entropy loss ˜ ˜ like for classification, then _gt_ ( _yi, yi_ ) = _yi −_ sigmoid( _yi_ ) ˜ ˜ ˜ and _ht_ ( _yi, yi_ ) = _−_ sigmoid( _yi_ ) _·_ (1 _−_ sigmoid( _yi_ )) with sigmoid( _z_ ) = (1 + exp( _−z_ )) _[−]_[1] . 

**Optimal split.** Each tree _ft_ is recursively built from a root node to the leaves: each node splits a dataset _X_ 

in a left _IL_ and right child _IR_ given some split criterion _s_ . Each child with its remaining dataset is then the basis for the next subtree. The process stops until a stopping criterion, e.g. the maximal tree depth, is reached. An optima-preserving equivalent of the optimal split criterion for each split _s_ is derived as: 

**==> picture [130 x 82] intentionally omitted <==**

where _λ_ is the regularization parameter of Ω. 

**Optimal leaf value.** The leaves of _ft_ contain the tree’s prediction, which is derived with the Newton method as: 

**==> picture [232 x 43] intentionally omitted <==**

## **3.3 Individual Rényi filter** 

The conventional privacy accounting approach involves a worst-case analysis of the privacy loss, assuming the global sensitivity for all individuals. This results in an overly conservative estimation of the privacy loss, as often a data point with many similar data points in the dataset has an individual sensitivity that is smaller than the global sensitivity and only those data points which have few similar data points in the dataset utilize the full sensitivity. 

An individual Rényi filter [14], is a way to implement personalized privacy accounting. The Rényi filter measures privacy losses individually via individual Rényi Differential Privacy (cf. Def. 10) and guarantees that the privacy loss of no data point will surpass a predefined upper bound _ρ_ ( _α_ ). This enables a differentially private mechanism, comprising a composed sequence of mechanisms _M_ 1 _, M_ 2 _, ..., MT_ max , to execute as many rounds as desired using data points that have not expended their privacy budget, as long as the accounting for individual Rényi privacy losses is sound. 

**Definition 10** (Individual Rényi Differential Privacy) **.** Fix _n ∈_ N and a data point _xi_ . A randomized mechanism _M_ satisfies ( _α, ρ_ ( _α_ ))-individual Rényi differential privacy if for all neighboring datasets _X, X[′]_ that differ in _xi_ , denoted as _X ∼xi X[′]_ , and satisfy _|X|, |X[′] | ≤ n_ , it holds that _Dα_ ( _M_ ( _X_ ) _||M_ ( _X[′]_ ) _≤ ρ_ ( _α_ ) 

Alg. 1 shows the individual Rényi filter algorithm. The algorithm obtains the individual Rényi privacy loss of round _t_ for each data point _xi_ (line 3). Next, the algorithm filters out all the data points that have surpassed the predefined upper bound on the privacy loss _ρ_ ( _α_ ) (line 4) using the privacy filter from Thm. 11 and then continues execution of mechanism _Mt_ in round _t_ only on the active data points that have not been filtered out. Feldman and Zrnic [14] show that Alg. 1 satisfies ( _α, ρ_ ( _α_ ))-Rényi DP (cf. Thm. 12). 

**Theorem 11** (Rényi Privacy Filter, [14] Theorem 4.3) **.** _Let_ 

**==> picture [225 x 28] intentionally omitted <==**

4 

_where ρ_ ( _α_ ) _is the upper bound on the privacy loss and ρi_ ( _α_ ) _(i ∈{_ 1 _,_ 2 _, . . . , k}) the individual privacy loss of a data point for round i. Then Fα,ρ is a valid Rényi privacy filter._ 

**Algorithm 1:** Adaptive composition with individual privacy filtering (cf. [14, Algorithm 3]) 

**Input:** _D_ : dataset **:** ( _M_ 1 _, M_ 2 _, . . . , MT_ max ) : sequence of mechanisms **:** ˆ _α_ : Rényi DP parameter **:** _ρ_ (ˆ _α_ ) : upper bound on Rényi DP privacy loss **1 for** _t_ = 1 _to Tmax_ **do 2 for** _xi ∈ D_ **do 3** _ρ_[(] _t[i]_[)][(] _[α]_[) :=] sup _X∼xi X′ Dα_ ( _Mt_ ( _a_ 1 _, . . . , at−_ 1 _, X_ ) _||Mt_ ( _a_ 1 _, . . . , at−_ 1 _, X[′]_ )) **4** Determine active set _Dt_ = ( _xi_ : _Fα,ρ_ ˆ (ˆ _α_ )( _ρ_[(] 1 _[i]_[)] _[, ρ]_[(] 2 _[i]_[)] _[, . . . , ρ] t_[(] _[i]_[)][) =][ CONT][)] **5** For all _xi ∈ D_ , set _ρ_[(] _t[i]_[)] _←_ 0 if _xi ∈/ Dt_ **6** Compute _at_ = _Mt_ ( _a_ 1 _, . . . , at−_ 1 _, Dt_ ) **7 return** ( _a_ 1 _, a_ 2 _, . . . , aT_ max ) 

**Theorem 12** (Theorem 4.5 in [14]) **.** _Adaptive composition with individual Rényi filters ( Alg. 1) using the Rényi filter from Thm. 11 satisfies_ ( _α, ρ_ ( _α_ )) _-Rényi differential privacy._ 

The intuition for the proof of Thm. 12 is that, whether a data point _x_ is active in some round, does not depend on the rest of the input dataset but only on the outputs of prior rounds, which are known to the adversary. Once _x_ is excluded from training, it does not lose any more privacy, because the output of any round from there on looks the same whether _x_ is present in the dataset or not. 

## **3.4 Subsampled Rényi differential privacy** 

Privacy amplification by subsampling allows a stronger privacy guarantee when choosing the data points for a single training round randomly from the training dataset rather than training on a fixed subset of the training dataset or even on the whole training dataset. Privacy amplification by subsampling was first analyzed by Li et al. [22]. In this work, we utilize Poisson Subsampling [3] where each data point is chosen according to a Bernoulli experiment with probability _γ_ , to obtain a batch of training data. We utilize the bound of Thm. 14 by Zhu and Wang [34] for subsampled Rényi DP which demands that the to-be-subsampled mechanism satisfies a lower bound on its Pearson-Vajda _X[l]_ pseudo-divergence. 

**==> picture [233 x 61] intentionally omitted <==**

**Theorem 14** (Privacy Amplification by Subsampling, Theorem 8 in [34]) **.** _Let M be any randomized mechanism that obeys_ ( _α, ρ[′]_ ( _α_ )) _-Rényi differential privacy. Let γ be the_ 

_subsampling ratio and α ≥_ 2 _. Let M[P][γ]_ = _M ◦Pγ and Pγ generating a Poisson subsample with subsampling ratio γ. If for all neighboring datasets X ∼ X[′] and all odd_ 3 _≤ l ≤ α, DX l_ ( _M_ ( _X_ ) _||M_ ( _X[′]_ )) _≥_ 0 _then M[P][γ] is tightly_ ( _α, ρ_ ( _α_ )) _- Rényi differentially private with ρ_ ( _α_ ) = _α−_ 1 1[log] (1 _−_ � _α γ_ ) _[α][−]_[1] ( _αγ − γ_ + 1) +[�] _[α] l_ =2 � _l_ �(1 _− γ_ ) _[α][−][l] γ[l] e_[(] _[l][−]_[1)] _[·][ρ][′]_[(] _[l]_[)] _._ � 

## **4 Problem statement & related work** 

Our task is to build a differentially private gradient boosted decision tree ensemble (GBDT). As indicated in the preliminaries, a GBDT has two primary data-dependent parts: splits and leaves. In prior work, the splits are selected data-independent, e.g. a random split selection, whereas the leaves are built with an additive noise mechanism. S-BDT uses both techniques as a foundation. 

## **4.1 Random split selection** 

A decision tree ensemble with randomized splits can yield good utility [7] and limits the privacy leakage to learning leaf-values. Random splits are constructed by randomly selecting a feature of the dataset and a value for that feature as a split. For a numerical feature _i_ , we assume a fixed feature range ( _v_ min[(] _[i]_[)] _[, v]_ max[(] _[i]_[)][)][for][sampling.] Bojarski et al. [7] propose to use random splits for binary classification using random forests: As the trees in the ensemble are random, most trees do not improve the decision to which class a data point belongs. This yields an almost even distribution between trees predicting class ‘0’ and class ‘1’. With high probability, however, a few random trees do improve the class-prediction, which suffices to tilt the prediction of the ensemble toward the right class. 

Nori et al. [28] propose _cyclical feature interaction_ : Split in all nodes of each tree _t_ on one feature _i ∈{_ 1 _, . . . , m}_ only and the next tree splits on the next feature ( _i_ = _t_ mod _m_ ). This technique boosts the performance in a DP setting. 

Maddock et al. [24] also investigate random splits and propose different candidate selections: one variant is based on candidates that are chosen equidistantly from a predefined split candidate set. This prevents too fine-grained splits but is also not flexible. Another variant is called _iterative Hessian_ which refines an initial equidistant split candidate set during ensemble training using tree-specific information: the aggregated Hessian as used in the leaf. For each split candidate, the aggregated Hessian is calculated: small values indicate data point absence which is handled by merging split candidates while large values indicate a large data point density which is handled by further subdividing this split candidate. A differentially private aggregated Hessian is obtained the same as in the leaf. This method refines the split candidates for the first _s_ training rounds and increases the privacy budget for our algorithm as follows. If S-BDT without split refinement is ( _α, ρ_ 1( _α_ ))-RDP, then S-BDT with split refinement is ( _α, ρ_ 1( _α_ ) + _sρ_ 2( _α_ ))-RDP where _ρ_ 2( _α_ ) is the RDP bound for only releasing the Hessian. The bound is derived via sequential composition (cf. Thm. 5) and because the split refinement uses all training data. Our experiments show that the split refinement does not improve the utility of S-BDT. 

5 

## **4.2 Leaf Noising** 

The leaf value _V_ is stated in Eq. (1). Prior work [23] proposes to additively noise each leaf value _V_ with Laplace noise proportional to the sensitivity _s_ of the leaf value: _V_ + Lap(0 _,[s] /ε_ leaf). Since each data point is only in one leaf of a tree, parallel composition applies which means that the privacy budget for each leaf is the same as for all leaves of a tree. While they provide a loose sensitivity to the leaf value, Maddock et al. [24] propose to Gaussian noise the numerator, i.e. gradient sum, and denominator, i.e. Hessian sum, individually. Thus, we only need to bound each sensitivity individually which is significantly less loose. For classification and due to its loss function, each gradient _g_ is naturally bounded between [ _−_ 1 _,_ 1] and each Hessian _h_ between [0 _,_ 0 _._ 25]. For regression, each gradient is unbounded but each Hessian is exactly 1. To accommodate both settings, we clamp each gradient _g_ by _g[∗]_ and each Hessian _h_ by _h[∗]_ which can even for classification boost the utility-privacy tradeoff as the noise scales with the clipping bound. Concretely, the sensitivity of a clamped version of the gradient sum �( _xi,yi_ ) _∈I_ Leaf[clamp][(] _[g][t]_[(] _[ϕ][t]_[(] _[x][i]_[)] _[, y][i]_[)] _[,][ −][g][∗][, g][∗]_[)][is][bounded][by] a fixed _g[∗]_ where clamp( _z, −g[∗] , g[∗]_ ) clips input _z_ to [ _−g[∗] , g[∗]_ ] and _I_ Leaf are the data points within a leaf. Similarly, the sensitivity of a clamped version of the Hessian sum �( _xi,yi_ ) _∈I_ Leaf[clamp][(] _[h][t]_[(] _[ϕ][t]_[(] _[x][i]_[)] _[, y][i]_[)] _[,][ −][h][∗][, h][∗]_[)][is][bounded][by] _h[∗]_ . 

## **4.3 Distributed learning: Batched Updates** 

To reduce the communication cost in distributed GBDT training, Maddock et al. [24] propose batched updates: A technique that only syncs the prediction updates among the users and thus updates the gradient and Hessian every _B_ batches of the _T_ total rounds. For syncing, _B_ many prediction updates are averaged. This technique effectively reduces the communication cost from _O_ ( _T_ ) to _O_ ( _[T] /B_ ) as only _[T] /B_ many trees are trained sequentially and the others in parallel like in a random forest. Utility-wise, less information is used, yet Maddock et al. [24] suggest in their experiments no significant utility loss and a slight utility advantage in the high-privacy regime ( _ε ≤_ 0 _._ 1) for small batches. Privacy-wise, the accounting remains the same as the same number of trees are released. 

## **4.4 Discussion of closely related work** 

**DP GBDT learners.** Maddock et al. [24] establish GBDTs as a promising direction for differentially private machine learning. We contribute significantly to the utilityprivacy tradeoff by privacy amplification by _subsampling_ and our _leaf-balanced noise_ . We pose a solution to learning on a stream of non-IID data using Rényi filters. 

- Subsampling does not improve Maddock et al. [24] without our tight bounds in Lem. 15 as it either leads to an untight generic bounds (e.g. a factor 15 worse in one of our runs) or requires univariate noise (due to a missing condition that we prove). As we add 2-dimensional noise per leaf in Newton Boosting, a subsampled Maddock et al. [24] would need to fall back to an inefficient gradient boosting like in [23] where we add noise per leaf proportional to _O_ ( _g[∗]_ ) instead of _O_ ( _[g][∗] /n_ ) with the gra- 

dient clipping bound _g[∗]_ and the number of data points in a leaf _n_ . 

- Leaf-balanced noise does not improve Maddock et al. [24] without our tight bounds in Thm. 16 as prior mechanisms like MVG [10] do not provide tight privacy bounds (e.g. RDP). As in MVG no RDP bounds are provided, other improvements like subsampling and individual Rényi filters remain an open challenge. 

- Maddock et al. [24] do not investigate individual Rényi filters, training an initial score, or learning on a stream of non-IID data. 

- For distributed learning, we use similar techniques as in [24]. 

**Subsampling for individual RDP.** The work by Yu et al. [33] applies individual privacy accounting to DPSGD and shows that individual RDP works seamlessly with privacy amplification by subsampling. Their work uses the moments accountant [1] which can be applied to spherical multivariate Gaussians and needs numerical evaluation that can suffer from imprecision compared to an analytical bound. In this work, we _prove_ a tight RDP bound for subsampling by utilizing the analytical bound by Zhu and Wang [34]. We also generalize the condition required for using the bound by Zhu and Wang [34] to a novel condition which only states that the RDP bound of the to-be-subsampled mechanism has to be linear in RDP _α_ . Using our novel condition, we can precisely evaluate the bound of Zhu and Wang [34] even for our leaf-balanced noise which resembles a non-spherical multivariate Gaussian. 

The original individual Rényi Filter work [14] does not cover subsampled RDP bounds. A follow-up work by Koskela et al. [21] applies individual Gaussian DP to a non-subsampled DP-GD; however, their work does cover a subsampled DP-SGD by a numerical implementation of a privacy loss accountant instead of an analytical bound as we have. 

**Leaf-balanced noise.** Prior work [24] on DP GBDTs does not offer the noise weighting of our leaf-balanced noise. There is work on using non-spherical multivariate Gaussian noise for DP: MVG [10, Theorem 3] proposes differential privacy accounting of matrix-variate Gaussian noise. For that, they bound the product of the norms of the singular values of the inversed covariance matrices proportional to _O_ ( _ε_ ) and _O_ ( _ln_[2] _δ_ ). This offers a more general setting, as S-BDT only handles diagonal multi-variate Gaussian noise[2] , yet our work provides tight RDP accounting which is essential for applying the tight subsampling bounds (cf. Sec. 5.1) as well as individual RDP accounting (cf. Thm. 16), which has not been covered by MVG. 

Another work [25, Lemma 1] proposes a technique that noises the ratio of gradient sum to subsampled batch size (in our case the Hessian sum) once instead of each sum individually. For that, they lower bound the subsampled batch size. However, this only works if the subsampled batches have a similar size. In the case of GBDTs, we have a widely varying Hessian sum, where a fixed lower bound would heavily deteriorate utility. In fact, Li et al. [23] assume such a scenario of a subsampled batch size of 1 (better: 0) in each leaf. 

> 2Our leaf-balanced noise could be generalized if the covariance matrix is public, since rotations preserve distances and thus the sensitivity remains unchanged. 

6 

## **5 Tight Individual RDP for subsampled non-spherical multivariate Noise** 

In these sections, we present novel technical components for individual Rényi DP that might be of independent interest. 

## **5.1 Exact RDP bounds for Poisson subsampling** 

We employ subsampling to boost the utility-privacy tradeoff. To also facilitate an individual Rényi filter we need subsampling bounds in individual Rényi DP accounting, for which prior work [33] has shown that an individual Rényi filter works seamlessly with subsampled individual Rényi DP. 

For subsampled RDP, there exists a generic analytical bound by Zhu and Wang [34] (cf. Thm. 14) which is tight if the Pearson-Vajda _X[l]_ pseudo-divergence _DX l_ ( _M_ ( _D_ 0) _||M_ ( _D_ 1)) is semi-positive for all neighboring datasets _D_ 0 _∼ D_ 1, the to-be-subsampled mechanism _M_ and all odd 3 _≤ l ≤ α_ . When _γ α e[ρ]_[(] _[α]_[)] _≪_ 1 the tight bound and the general upper bound [34, Theorem 5] match up to a multiplicative factor of 1 + _O_ ( _γ α e[ρ]_[(] _[α]_[)] ) for subsampling ratio _γ_ and Rényi-DP bound of the to-be-subsampled mechanism ( _α, ρ_ ( _α_ )). In all other cases, the bounds match up to an additive factor of[lo] _α_[g] _−_[(] 1[3][)][.][For][our][best-performing] hyperparameter setting of S-BDT on the Abalone regression dataset for _ε_ = 0 _._ 1, we observe an improvement in _ρ_ ( _α_ ) from 0 _._ 7443 to 0 _._ 04705 for _α_ = 241 when using the tight bound compared to the general upper bound. 

Zhu and Wang [34, Theorem 17] provide a proof for one- 

dimensional Gaussians that enables usage of the tight subsampling bound. Consequently, this proof does not apply to more general mechanisms, including mechanisms using our leaf-balanced noise, a non-spherical multivariate Gaussian. We provide in Lem. 15 a general proof of the Pearson-Vajda _X[l]_ pseudo-divergence condition with one notable requirement: To apply the tight subsampling bound of Thm. 14 [34], the to-be-subsampled mechanism _M_ has to have a RDP bound _ρ_ ( _α_ ) that is linear in _α_ and semi-positive. This holds for variants of the Gaussian mechanism like multi-variate or non-spherical Gaussian, cf. Thm. 16, but not Laplace or randomized response (cf. [26, Table 2]). 

**Lemma 15.** _Let M be the to-be-subsampled mechanism that satisfies_ ( _α, ρ_ ( _α_ )) _-Rényi DP and X ∼ X[′] two neighboring datasets. If ρ_ ( _α_ ) = _α · t for any t ≥_ 0 _, then the Pearson-Vajda X[α] pseudo-divergence of M is semi-positive, i.e. DX[α]_ ( _M_ ( _X_ ) _||M_ ( _X[′]_ )) _≥_ 0 _, for all odd α ≥_ 1 _._ 

_Proof._ By induction, we show that for all odd _α ≥_ 1, _M_ satisfies the condition for any neighboring datasets _X ∼ X[′]_ . Let _o_ be an observation of _M_ and _pM_ ( _X_ ) _, pM_ ( _X′_ ) be the densities of _M_ ( _X_ ) _, M_ ( _X[′]_ ). 

## **Base case (** _α_ = 1 **):** 

**==> picture [191 x 53] intentionally omitted <==**

**Inductive step (** _α −_ 2 _→ α_ **):** Note that by rewriting the Rényi divergence we have 

**==> picture [220 x 18] intentionally omitted <==**

Using this equality and the binomial identity we get 

**==> picture [234 x 49] intentionally omitted <==**

We investigate the partial derivative in _t_ of the PearsonVajda _X[α]_ pseudo-divergence: 

**==> picture [192 x 49] intentionally omitted <==**

**==> picture [232 x 12] intentionally omitted <==**

**==> picture [228 x 68] intentionally omitted <==**

Using the binomial identity we get 

**==> picture [163 x 23] intentionally omitted <==**

By Zhu and Wang [34, Lemma 16] we lower bound 

**==> picture [164 x 49] intentionally omitted <==**

Since the partial derivative in _t_ is semi-positive and for _t_ = 0 we have E _M_ ( _X′_ )�� _MM_ (( _XX[′]_ )) _[−]_[1] � _α_ � = 0, we conclude that E _M_ ( _X′_ )�� _MM_ (( _XX[′]_ )) _[−]_[1] � _α_ � _≥_ 0 _∀t≥_ 0. Thus by induction, we conclude that E _M_ ( _X′_ )�� _MM_ (( _XX[′]_ )) _[−]_[1] � _α_ � _≥_ 0 _∀t≥_ 0 _∀α≥_ 1 _,α_ odd. If _ρ_ ( _α_ ) = _α · t_ then the Pearson-Vajda pseudodivergence precondition of Thm. 14 is fulfilled, thus the conclusion of Thm. 14 applies, i.e. the subsampled mechanism is RDP with _ρ_ ( _α_ ) = _α−_ 1 1[log] �(1 _−γ_ ) _[α][−]_[1] ( _αγ −γ_ +1)+[�] _[α] l_ =2 � _αl_ �(1 _− γ_ ) _[α][−][l] γ[l] e_[(] _[l][−]_[1)] _[·][ρ][′]_[(] _[l]_[)][�] . 

## **5.2 Individual RDP bound for Leafbalanced noise** 

To release the value of a leaf in GBDT in a differentially private manner, we consider the function _f_ ( _X_ ) = ( _w, u_ ) for an arbitrary dataset _X_ . _f_ clips the individual gradients and Hessians from _X_ to bound the sensitivities and computes the sum of gradients _u_ and the sum of Hessians _w_ . To preserve differential privacy, Gaussian noise is calibrated to the privacy budget and the clipping bound and applied to the output of _f_ . The leaf value is then constructed by ˜ post-processing on the released noisy values, _u/_ ˜ _w_ . 

7 

When noising _f_ , we utilize leaf-balanced non-spherical multivariate Gaussian noise with covariance matrix 

**==> picture [125 x 18] intentionally omitted <==**

with _r_ 1 _, r_ 2 satisfying _r_ 1 + _r_ 2 = 1. We sample _Y ∼N_ (0 _,_ Σ) and then release _f_ ( _X_ )+ _Y_ = ( _w, u_ )+ _Y_ . Here, _r_ 1 _, r_ 2 allow us to better calibrate the amount of noise that is applied to the two values _w_ and _u_ . 

We believe that our adaptation of the Gaussian mechanism with leaf-balanced noise is of independent interest, as its idea of finetuning the shares of noise over the dimensions of the output can be helpful for different functions even with an arbitrary number of output dimensions. We analytically derive individual Rényi DP bounds for this mechanism in Thm. 16. 

**Theorem 16** (Individual RDP of Gaussian Mechanism with leaf-balanced non-spherical noise) **.** _Let f_ : _X →_ R _[D] denote a function from an arbitrary input X ∈X to a set of scalars. Let the function f projected to its d-th output have bounded sensitivity sd ∈_ R _and bound individual sensitivity s_[(] _d[i]_[)] _∈_ R _. Let Y ∼N_ ( **0** _,_ Σ) _be a random variable of non-spherical multivariate Gaussian noise with covariance matrix_ Σ := _D[−]_[1] _·_ diag( _r_ 1 _[−]_[1] _[s]_ 1[2] _[σ]_[2] _[, . . . , r] D[−]_[1] _[s] D_[2] _[σ]_[2][)] _[where][a] given variance σ_[2] _∈_ R[+] _is weighted in each dimension d individually by s_[2] _d[and][r][d][∈]_[R][+] _[such][that]_[�] _[D] d_ =1 _[r][d]_[= 1] _[.][Then] the Gaussian mechanism M_ ( _X_ ) _�→{ f_ ( _X_ ) _d_ + _Yd }[D] d_ =1 _[sat-] isfies_ ( _α, ρ_ ( _α_ )) _-individual RDP for xi ∈ X with ρ_ ( _α_ ) = _α ·_ 2 _Dσ_[2] _[·]_[ �] _[D] d_ =1 _rd·_ ( _ss_[2] _d_[(] _d[i]_[)] )[2] _._ 

> _Proof._ We prove ( _α, ρ_ ( _α_ ))-individual RDP as follows: (1) We derive the privacy loss distribution (PLD) of the _d_ - th output of _M_ ( _X_ ). (2) We use that the PLD of a _D_ - fold sequential composition of _M_ is the same as a _D_ - fold convolution of the 1-dimensional PLD. (3) We show ( _α, ρ_ ( _α_ ))-individual RDP via the convoluted PLD, which constitutes a Gaussian distribution. 

(1) Let _X ∼xi X[′]_ be neighboring datasets differing in _xi_ . Let _o_ be an atomic event, and _LM_ ( _X_ ) _/M_ ( _X′_ )( _o_ ) := ln( Pr[[Pr][[] _M[M]_ ([(] _X[X][′]_[)] )=[=] _[o] o_[]] ][)][ be the privacy loss of mechanism] _[ M]_[.][Then,] the privacy loss distribution _ω_ ( _y_ ) on support _y ∈ Y_ := � _o_ � _LM_ ( _X_ ) _/M_ ( _X′_ )( _o_ ) � as defined in Sommer et al. [31, Definition 2] resembles a probability distribution over the privacy losses _LM_ ( _X_ ) _/M_ ( _X′_ ). Prior work has shown that if there are worst-case distributions, there is a PLD that fully describes the leakage of any mechanism. Any additive mechanism for a sensitivity-bounded query has worst-case distributions: a pair of Gaussians of which one is shifted by the sensitivity. Since _M_ is the Gaussian mechanism, we get a closed form for _ω[d]_ on the _d_ -th output of _M_ for _X ∼xi X[′]_ [31, Lemma 11]: 

**==> picture [135 x 20] intentionally omitted <==**

Note: _ω[d]_ is the same for both privacy losses _LM_ ( _X_ ) _/M_ ( _X′_ ) and _LM_ ( _X′_ ) _/M_ ( _X_ ) since a Gaussian is symmetric. 

(2) One characteristic of a PLD is that a _D_ -fold sequential composition corresponds to a _D_ -fold convolution of _ω_ [31, Theorem 1]. Hence, the convolution of two Gaussians is Gaussian again: _N_ ( _µx, σx_[2][) +] _[ N]_[(] _[µ] y[, σ] y_[2][)][=] _N_ ( _µx_ + _µy, σx_[2][+] _[σ] y_[2][)] _[,][∀] µx,µy ,σx,σy_[.] Thus, we have the following closed form for _ω_ on the _D_ -dimensional output 

of _M_ ( _X_ ): 

**==> picture [205 x 56] intentionally omitted <==**

(3) We convert the convoluted PLD _ω_ to an ( _α, ρ_ ( _α_ ))-RDP bound as follows: 

**==> picture [143 x 11] intentionally omitted <==**

by [31, Lemma 8] we get 

**==> picture [117 x 18] intentionally omitted <==**

by the moment generating function: 

**==> picture [178 x 14] intentionally omitted <==**

**==> picture [101 x 11] intentionally omitted <==**

**==> picture [127 x 30] intentionally omitted <==**

**Corollary 17.** _The Gaussian mechanism with leafbalanced non-spherical noise satisfies_ ( _α, α ·_ 2 _Dσ_[2][)] _[-RDP.]_ 

_Proof._ The corollary follows directly from Thm. 16 for a challenge data point with worst-case sensitivity _sd_ . 

We apply Thm. 16 to S-BDT in Cor. 21 where we have a bivariate case of non-spherical Gaussian noise. In this way, S-BDT utilizes a leaf-balanced noise. We experimentally show (cf. Sec. 8.1 and Sec. 8.3) that our leaf-balanced noise with _r_ 1 = _r_ 2 performs better than the standard spherical Gaussian mechanism where _r_ 1 = _r_ 2 = 0 _._ 5. 

## **6 S-BDT: Tighter DP GBDT** 

In this section, we detail our S-BDT algorithm and our improvements to DP GBDTs: we start with Alg. 2 as the starting point of S-BDT training (cf. overview Sec. 6.1) and continue with Alg. 3 as our _differentially private initial score_ (cf. Sec. 6.2) and with Alg. 4 as the training routine of a single tree with _subsampling_ and random splits (cf. Sec. 6.3). Every single tree utilizes our _leaf-balanced noise_ which we describe together with the leaf value calculation in Alg. 5 (cf. Sec. 6.4). Then, we finish by presenting the individual Rényi DP accountant in Alg. 6 (cf. Sec. 6.5) and our two generalizations: Learning on a stream of non-IID data (cf. Sec. 6.6) and scalable distributed learning (cf. Sec. 6.7). 

## **6.1 Algorithm overview** 

S-BDT trains a GBDT ensemble as follows (cf. Alg. 2). After initialization of the individual Rényi DP accountant (line 2), we compute the initial score from the labels in the dataset using Alg. 3 (line 3) and add it to the ensemble _E_ . S-BDT then runs _T_ max = _T_ regular + _T_ extra training rounds where in each round we 1) apply the individual Rényi 

8 

**Algorithm 2:** TrainSBDT : Train a DP GBDT ensemble 

||**Input:** _D_ : training data, _γ_ : subsampling ratio|**Input:** _D_ : training data, _γ_ : subsampling ratio|**Input:** _D_ : training data, _γ_ : subsampling ratio|**Input:** _D_ : training data, _γ_ : subsampling ratio|
|---|---|---|---|---|
||**:**_ε_init : privacy budget for initial score||||
||**:**(_ε_trees_, δ_trees) : DP parameters for training||||
||of trees||||
||**:**(_r_1_, r_2) : noise weights for leaf value||||
||**:**(_g∗, h∗, m∗_) : gradient/Hessian/label||||
||clipping bound||||
||**:**_λ, β_ : regularization parameters||||
||**:**(_T_regular_, T_extra) : number of regular and||||
||extra rounds||||
||**:**_α_max : largest _α_ to test in Rényi||DP, _d_|:|
||depth of trees||||
|**1**|_T_max =_T_regular+_T_extra||||
|**2**|ˆ_α, σ_2<br>leaf_, ρ_(ˆ_α_) = `Initialize(`_α_max_,_||||
||(_εtrees, δtrees_)_, εinit, γ_`)`||||
|**3**|init0 =`DPInitScore(`_D, m∗, εinit_`)`||||
|**4**|_E_ = (init0)||||
|**5 **|**for** _t_= 1 _to Tmax_ **do**||||
|**6**|**for** _i_= 1 _to |D|_ **do**||||
|**7**|_ρ_(_i_)<br>_t_ (_α_) :=_aγ_(_α,_<br>_α_<br>_σ_2<br>leaf _·_<br>�<br>_r_1_·|hi|_2<br>(_h∗_)2|+|_r_2_·|gi|_2<br>(_g∗_)2|�<br>);|
||`// by Thm. 19 (for RDP: Cor. 20)`||||
|**8**|_Dt_ = (_xi_ :_F_ˆ_α,ρ_(ˆ_α_)(_ρ_(_i_)<br>1 _, . . . , ρ_(_i_)<br>_t_ ) =|CONT);|||
||`// by Thm. 11`||||
|**9**|tree_t_ = `TrainSingleTree(`_Dt, d,_||||
||_σ_2<br>_leaf, g∗, h∗,_(_r_1_, r_2)_, λ, β, E_`)`||||
|**10**|_E_ = (init0_,_tree1_, ...,_tree_t_)||||
|**11 **|**return** E||||



filter by computing individual RDP bounds for all data points using Thm. 19 (line 7) such that we filter out those data points that have exceeded their privacy budget using Thm. 11 (line 8) and 2) train a single tree on the filtered dataset using Alg. 4 (line 9) and add it to the ensemble _E_ . In Thm. 23 we show that ensemble training with S-BDT satisfies ( _α, ρ_ ( _α_ ))-RDP. 

## **6.2 Initial score** 

GBDT ensemble training needs an initial score so that the ensemble can then add trees to improve on the error of the initial base classifier that outputs the initial score. Prior works chose the initial score simply as 0 _._ 0, however, it can be beneficial (cf. Experiment Sec. 8.1 and 8.3) that the initial score gives a more meaningful starting point for the ensemble. We output the mean of labels from the dataset as the initial score and release this mean with the Laplace mechanism to preserve ( _α, ρ_ ( _α_ ))-Rényi DP, which we show in Thm. 18: 

**Theorem 18.** _Alg. 3 (_ _`DPInitialScore` ) with clipping bound m[∗] , DP approximated dataset size |D|priv and f_ ( _ε_ ) = log � 2 _αα−_ 1[exp] �( _α −_ 1) _· ε_ � + 2 _αα−−_ 11[exp] � _− α · ε_ �� _satisfies_ � _α, α−_ 1 1[(] _[f]_[(] _[ε][init]_[) +] _[ f]_[(] _[ε][ds]_[))] � _-RDP._ 

_Proof._ For the complete proof we refer to the appendix. It uses sequential composition and a generalization of Mironov [26, Proposition 6] for arbitrary sensitivity. 

**Description of algorithm.** 

**Algorithm 3:** DPInitialScore : Compute a DP initial score 

||**Input:** _D_ : training dataset, _m∗_: clipping bound|
|---|---|
||on labels|
||**:**_ε_init : DP privacy budget for initial score|
||**:**_ε_ds = 0_._005: DP privacy budget for dataset|
||size|
|**1**|_|D|_priv =_|D|_+`Laplace(`0_,_1_/εds_`)`|
|**2 **|**if** _regression_ **then**|
|**3**|_M_ =<br>1<br>_|D|_priv<br>�<br>_yi∈D_ `clamp(`_yi, −m∗, m∗_`)`|
|**4**|_M_priv =_M_ +`Laplace(`0_, m∗/_(_|D|priv · εinit_)`)`|
|**5**|**return** _M_priv|
|**6 **|**else if** _classifcation_ **then**|
|**7**|_M_ =<br>1<br>_|D|_priv<br>�<br>_yi∈D_ `clamp(`_yi,_0_, m∗_`)`|
|**8**|_M_priv =_M_ +`Laplace(`0_, m∗/_(_|D|priv · εinit_)`)`|
|**9**|**return** ln(_M_priv_/_1_−M_priv)|



In Alg. 3, we release the dataset size in a DP manner using the Laplace mechanism with privacy budget _ε_ ds (line 1). We fix _ε_ ds = 0 _._ 005. We clamp the labels of all data points to a fixed range (line 3) to upper bound the influence of any data point on the initial score. Then we average the clamped labels (line 3) and add Laplace noise calibrated to the clipping bound _m[∗]_ , the privacy budget _ε_ init, and the DP approximated number of data points _|D|_ priv (line 4) to satisfy differential privacy. For classification, we rescale the noised mean with the logit which is the inverse of the sigmoid function. 

## **6.3 Training a single tree with subsampling** 

To boost the utility-privacy tradeoff, we tailor privacy amplification by subsampling which is well discussed in literature [1, 3, 22, 34] to GBDTs. For subsampling, we train each tree in GBDT only on a random subsample of the training data, where we select each training data point with probability _γ_ . An ( _ε, δ_ )-DP mechanism on all training data becomes an ( _O_ ( _γε_ ) _, γδ_ )-DP mechanism on the subsampled data [3, Theorem 8]. 

**Description of algorithm.** Alg. 4 describes the training of a single tree. First, we generate a subsample of the training data via Poisson subsampling (line 1). Second, we sample a random tree of full depth _d_ (line 2). Third, we assign all leaf values using differentially private Alg. 5 (lines 4-5). 

We experimentally show an improvement in the utilityprivacy tradeoff through subsampling for S-BDT in Sec. 8.1 and Sec. 8.3. 

**Rényi DP proof.** 

**Theorem 19.** _Let r_ 1 _, r_ 2 _be defined as in Thm. 16. Let σleaf_[2] _[be][the][unweighted][variance][of][the][leaf][Gaussian.][Let] aγ_ : N _×_ R _�→_ R _denote the privacy amplification of Thm. 14 with subsampling ratio γ. Then, for data point xi with gradient gi and Hessian hi, Alg. 4 (_ _`TrainSingleTree` ) satisfies_ ( _α, aγ_ ( _α, α ·_ 2 _σ_ 2 _leaf_[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2][+] _[r]_[2] ( _g[·][|][∗][g]_ ) _[i]_[2] _[|]_[2] �)) _-individual RDP._ 

_Proof._ For the complete proof we refer to the appendix. The proof uses the individual RDP bound of a leaf _ρ_ ( _α_ ) = _α · t_ = _α ·_ 2 _σ_ 2leaf[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2] + _[r]_[2] ( _g[·][|][∗][g]_ ) _[i]_[2] _[|]_[2] � (cf. Cor. 21) as 

9 

**Algorithm 4:** TrainSingleTree: Train a DP decision tree 

||**Input:** _D_ : training data, _d_ : depth of trees|
|---|---|
||**:**_σ_2<br>leaf : unweighted variance of Gaussian for|
||leaves|
||**:**(_g∗, h∗_) : clipping bound on gradients and|
||Hessians|
||**:**(_r_1_, r_2) : noise weights for leaf value|
||**:**_λ, β_ : regularization parameters|
||**:**_E_ : ensemble of trees up to round _t −_1|
|**1**|_DP_ =`PoissonSubsample(`_D, γ_`)`;|
|**2**|tree_t_ =`RandomTree` (d);<br>`// cf.`<br>`Sec. 4.1`|
|**3 **|**for each** _leaf l_ **in** _treet_ **do**|
|**4**|_v_ =`DPLeaf(`_l, DP, g∗, h∗, σ_2<br>_leaf,_(_r_1_, r_2)_, λ, β_`)`;|
|**5**|`SetLeaf` (tree_t, l, v_);|
|**6 **|**return** tree_t_;|



well as our novel condition for tight subsampling bounds (cf. Lem. 15), i.e. Alg. 4 ( `TrainSingleTree` ) when no subsampling is applied has a Pearson-Vajda _X[α]_ pseudodivergence of _DX[α]_ ( _M_ ( _X_ ) _||M_ ( _X[′]_ )) _≥_ 0 for all odd _α ≥_ 1. This allows us to use the tight individual RDP bound for subsampling by Zhu and Wang [34]. 

**Corollary 20.** _`TrainSingleTree` (cf. Alg. 4) satisfies_ ( _α, aγ_ ( _α, α/σleaf_ 2[))] _[-RDP.]_ 

_Proof._ The corollary follows directly from Thm. 19 with the worst-case sensitivities _g[∗]_ and _h[∗]_ . 

## **6.4 Leaf computation with leaf-balanced noise** 

**Description of algorithm.** S-BDT computes leaves for each tree in a GBDT ensemble using Alg. 5. Our _leaf-balanced_ noise uses the foundation of prior work as detailed in Sec. 4.2: We calculate and Gaussian noise the sum of clamped gradients and clamped Hessians. With clamping we bound the influence of each gradient _gi_ such that _|gi| ≤ g[∗]_ and each Hessian _hi_ such that _|hi| ≤ h[∗]_ . We release a differentially private leaf value by dividing the noised sum of gradients by the noised sum of Hessians. 

**Leaf-balanced noise.** Our leaf-balanced noise (Sec. 5.2) allows us to better calibrate the amount of noise that is applied to the sum of gradients and sum of Hessians (lines 3 and 2 in Alg. 5). We experimentally show (cf. Sec. 8.1 and Sec. 8.3), that our leaf-balanced noise with _r_ 1 = _r_ 2 performs better than the standard non-spherical multivariate Gaussian mechanism where _r_ 1 = _r_ 2 = 0 _._ 5. In our experiments, choosing the range 0 _._ 05 _≤ r_ 1 _≤_ 0 _._ 2 is most helpful which assigns more budget to the gradient sum (leaf numerator). 

**Rényi DP proof.** 

**Corollary 21.** _Alg. 5 (_ _`DPLeaf` ) satisfies_ ( _α, α ·_ 2 _σ_ 2 _leaf_[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2][+] _[r]_[2] ( _g[·][|][∗][g]_ ) _[i]_[2] _[|]_[2] �) _-individual RDP for data point xi with gradient gi and Hessian hi and with σleaf_[2] _[as][the][unweighted] variance of the leaf Gaussian._ 

_Proof._ For the complete proof we refer to the appendix. The proof directly follows from our leaf-balanced noise 

(cf. Thm. 16) which shows that the overall privacy budget remains the same if we weigh each dimension via _r_ 1 _, r_ 2 differently as long as _r_ 1 + _r_ 2 = 1. In `DPLeaf` we represent the leaf noising as adding a 2-dimensional Gaussian to a vector of gradient sum (leaf numerator) and Hessian sum (leaf denominator). Building this ratio is DP due to the post-processing theorem. 

**Corollary 22.** _Alg. 5 (_ _`DPLeaf` ) satisfies_ ( _α,[α] /σleaf_[2][)] _[-RDP.]_ 

_Proof._ The corollary follows directly from Cor. 21 with a challenge data point with worst-case sensitivities _g[∗]_ and _h[∗]_ . 

**Algorithm 5:** DPLeaf : Compute a DP leaf node 

||**Input:** _D_ :training data, _l_: leaf node identifer|
|---|---|
||**:**(_g ∈D, h ∈D_) : gradient/Hessian of data|
||points in _D_|
||**:**(_g∗, h∗_) : clipping bound on gradients and|
||Hessians|
||**:**(_r_1_, r_2) : noise weights for leaf value|
||**:**_λ, β_ : regularization parameters|
||**:**_σ_2<br>leaf : unweighted variance of Gaussian for|
||leaves|
|**1**|Let _Dl ⊆D_ be the set of data points in leaf _l_|
|**2**|_w_ = �<br>_h∈Dl_ `Clamp`(_h,_0_._0_, h∗_)|
|**3**|˜_w_ =_λ_+_w_+`Gauss(`0_._0_,_(_h∗_)2_σ_2<br>_leaf/_(2_· r_1)`)`|
|**4**|_u_= �<br>_g∈Dl_ `Clamp`(_g, −g∗, g∗_)|
|**5**|˜_u_=_u_+`Gauss(`0_._0_,_(_g∗_)2_σ_2<br>_leaf/_(2_· r_2)`)`|
|**6 **|**return** `Clamp(`_v_ = ˜_u/_˜_w, −β, β_`)`|



## **6.5 Rényi DP accounting** 

Our Rényi DP accountant picks an order _α_ ˆ to measure the privacy loss in Rényi DP, sets noise variance _σ_ leaf[2][that] is applied to the leaves and sets an upper bound on the individual privacy loss of our training, so that we can utilize an individual Rény filter. 

**Description of algorithm.** Our Rényi DP accountant (cf. Alg. 6) iterates over pairs of order _α_ and noise variance _σ_ leaf[2][and][uses][Cor.][20][(line][4)][to][compute][the] Rényi DP privacy leakage of order _α_ for subsampled training of a single tree when applying noise with variance _σ_ leaf[2] in the leaves. The accountant investigates orders of _α ≥_ 2 as for other _α_ the privacy amplification by subsampling of Thm. 14 does not apply. Next, the accountant applies RDP sequential composition (cf. Thm. 5) to account for the number of training rounds (line 5) and converts this RDP bound to a ( _ε, δ_ )-DP bound using Cor. 6. Over all orders _α_ and all noise variances _σ_ leaf[2][the][accountant][picks] the pair that satisfies the following condition: the converted ( _ε, δ_ ) bound is close to the user-specified privacy budget and the noise variance is minimal (line 8). 

The initialization returns (1) the order _α_ ˆ for which the Rényi DP accounting is done, (2) the noise variance _σ_ leaf[2] applied to the leaves and (3) an upper bound _ρ_ ( _α_ ˆ) for the individual privacy loss. The initialization is ( _ε_ init+ _ε_ trees _, δ_ )- DP for _δ_ = 5 _·_ 10 _[−]_[8] and _ε_ init and _ε_ trees being the DP budget of the initial score and the tree training. 

**Discussion.** Our accountant can measure subsampled Rényi DP, giving S-BDT a privacy amplification by subsampling. This sets our accounting apart from the 

10 

**Algorithm 6:** Initialize Rényi DP accountant 

||**Input:** _α_`max` : largest _α_ to test|**Input:** _α_`max` : largest _α_ to test|**Input:** _α_`max` : largest _α_ to test|in|RDP, _γ_|:|||
|---|---|---|---|---|---|---|---|---|
||||subsampling ratio||||||
||||**:**(_ε_trees_, δ_trees) : DP parameters for|||training|||
||||of trees||||||
||||**:**_ε_init : privacy budget for||DP initial||score||
|**1**|_T_ =||()||||||
|**2 **|**for** _α_= 2 _to αmax_ **do**||||||||
|**3**||**for** _σ_2<br>_leaf in_ (0_._0_,_1000_._0] **do**|||||||
|**4**|||_ρ_subsampled-tree =_aγ_(_α,_<br>_α_<br>_σ_2<br>leaf );||||`// Use`||
||||`Cor. 20`||||||
|**5**|||_ρ_(_α_) =_T_regular_· ρ_subsampled-tree;||||`// Use`||
||||`Thm. 5`||||||
|**6**|||_ε′_<br>trees =_ρ_(_α_) + log _α−_1<br>_α_<br>_−_log_δ_trees+log <br>_α−_1||||_α_|;|
||||`// Cor. 6`||||||
|**7**|||`Append(`_T ,_(_α, σ_2<br>_leaf, ρ_(_α_)_, ε′_<br>_trees_)`)`||||||
||||||||||
|**8**|Pick||(ˆ_α, σ_2<br>leaf_, ρ_(ˆ_α_)_, ε′_<br>trees) from|_T_|so that _σ_2<br>leaf is||||
||smallest and _ε′_<br>trees is close to _ε_trees||||||||
|**9**|Report _ε_init+_ε′_<br>trees to user||||||||
|**10 **|**return** ˆ_α, σ_2<br>leaf_, ρ_(ˆ_α_)||||||||



accounting of prior works that utilized parallel and sequential composition for DP. Our accountant is generic: the RDP bound of any mechanism can be plugged in to obtain an accurate and robust subsampled RDP accounting. 

## **6.6 Generalizing S-BDT: Stream of nonIID data** 

We consider learning a stream of non-IID data, where new data arrives over time, with DP GBDT. Here, we want to incorporate training data that arrives later and also not forget about formerly seen data points. Under privacy, however, training again with formerly seen data points poses the risk of additional privacy leakage. Privacy would hold if the GBDT model simply forgets about formerly seen data points, but this can hurt model performance. 

Tailoring an individual Rényi filter to DP GBDT training enables individual privacy accounting, which in turn enables training for an arbitrary amount of rounds with only those data points that still have privacy budget left. This enables S-BDT to effectively learn on streams of nonIID data: S-BDT learns with newly arrived data while also including data points that arrived in the past. 

We experimentally show (cf. Sec. 8.4) that when learning a stream of non-IID data, tailoring an individual Rényi filter to S-BDT can significantly improve the model performance. We compare to a naïve approach for stream learning where the formerly seen data points are discarded once they have been used for the amount of training rounds that the privacy budget has been accounted for. 

## **6.7 Scalable distributed learning** 

We extend S-BDT to work in a distributed learning setting, where multiple users collaboratively train a model, while the private training datasets are not directly shared. This applies to the field of medical studies where hospitals want to keep sensitive patient data at their hospital but want to train a machine learning model collaboratively. Our extension of S-BDT to distributed learning uses similar techniques as Maddock et al. [24]. For the detailed 

algorithm, we refer to the appendix. In summary: 

Every user receives the training hyperparameters from a secure bulletin board and sets up the accounting. For the initial classifier, the user computes DP releases of the sum of labels and the dataset size. All users synchronize and invoke `SecureAggregation` [8, 6] with fixed precision, for aggregating the label sum first and then dataset size. The initial score is then built by dividing the aggregated label sum by the overall dataset size and added to the ensemble. The user commences _T_ `max` rounds of training. Each round, the user updates the individual RDP privacy losses for all its data points and filters out data points that have expended their privacy budget. Next, the user locally initializes a tree for the current round. All user synchronize and utilize public uniform sampling [30, Protocol 1] to collaboratively and verifiably sample uniformly random features and feature values for the tree splits. 

The user locally only adjusts the leaves of the tree with its local share of sensitive data. All users synchronize again to utilize `SecureAggregation` with fixed precision for collaboratively generating leaf values for the tree. `SecureAggregation` is used for aggregating the gradient sum and the Hessian sum of all leaves, then the user sets the leaf values of its local tree and finally adds this tree to the ensemble. 

## **7 S-BDT is differentially private** 

**Theorem 23** (Main theorem (informal)) **.** _S-BDT is_ ( _α, ρ_ ( _α_ ) _)-Rényi differentially private._ 

_Proof._ Let _X_ be the training dataset, _T_ regular the number regular trees without extra trees due to a Rényi filter, _m[∗]_ the clipping bound on the labels, and _ε_ init _, ε_ ds the privacy budgets used by the DP initial score. Let _aγ_ : N _×_ R _�→_ R denote the privacy amplification of Thm. 14 _α_ with subsampling ratio _γ_ and _f_ ( _ε_ ) = log � 2 _α−_ 1[exp] �( _α −_ 1) _· ε_ � + 2 _αα−−_ 11[exp] � _− α · ε_ ��. We then have 

**==> picture [217 x 18] intentionally omitted <==**

S-BDT consists of the differentially private initial score mechanism, and then multiple training rounds, each outputting a single tree. So the proof decomposes S-BDT into the initial score step and the training rounds via sequential composition for Rényi DP. We can then separately bound the Rényi divergences for the initial score and the training rounds. 

For the initial score, we derive an analytical ( _α, ρ_ init( _α_ ))Rényi DP bound in Thm. 18, using a generalization of the Laplace mechanism to arbitrary sensitivity, as the sensitivity of our unnoised initial score depends on the DP approximated dataset size _|D|_ priv, the privacy budget for the initial score _ε_ init and the clipping bound _m[∗]_ on the labels from dataset _D_ . 

Next, we obtain a Rényi DP bound for the training rounds. Here is where the individual Rényi filter comes into play: A single training round consists of the individual Rényi filter application, i.e. computing the individual Rényi DP privacy losses via our analytical bound Thm. 19 (for RDP: Cor. 20) then filtering out those data points that have fully expended their privacy budget, before finally running the training round. In this setup, the individual Rényi filter (cf. Thm. 12) guarantees that an arbitrary amount of 

11 

training rounds satisfies an a priori ( _α, ρ_ training( _α_ ))-Rényi DP bound, as long as the individual Rényi DP accounting is sound. Combining the two analyses, we get that the output of S-BDT satisfies ( _α, ρ_ init( _α_ ) + _ρ_ training( _α_ )). For the complete proof we refer to the appendix. 

## **8 Empirical evaluation** 

We evaluate our differentially private gradient boosted decision trees ensemble learner S-BDT (cf. Sec. 6) as follows. In Sec. 8.1 we compare S-BDT against the state of the art (SOTA) by Maddock et al. [24]. In Sec. 8.3 we perform ablation studies of our improvements in S-BDT. We answer the following research questions: 

**(RQ1)** _Is S-BDT improving on the current SOTA [24]?_ In Sec. 8.1 we find that S-BDT significantly improves on the SOTA, saving 50% in terms of _ε_ for _ε ≤_ 0 _._ 5 on the regression dataset and 30% for _ε ≤_ 0 _._ 08 and _ε ≤_ 0 _._ 03 respectively on the classification datasets for the same utility (AUC/RMSE). 

**(RQ2)** _Under privacy, are our contributions leaf-balanced noise (cf. Sec. 6.4), subsampling (cf. Sec. 6.3), and DP initial score (cf. Sec. 6.2) improving the performance of S-BDT?_ In Sec. 8.3 we show leaf-balanced noise and subsampling consistently improve the performance of S-BDT. DP initial score improves utility on the regression dataset. 

**(RQ3)** _Do individual Rényi filters [14] improve the performance of S-BDT when learning a stream of non-IID data?_ In Sec. 8.4 we show that an individual Rényi filter can significantly improve the performance of S-BDT when learning a stream of non-IID data. 

**Sensitive Datasets.** We use three datasets for our experiments: Abalone, Adult and Spambase. Abalone [27] is a regression dataset containing 4 _,_ 177 data points. Given eight numerical attributes, e.g. sex, length, and diameter of an abalone, the task is to predict its age. Adult [5] is a binary classification dataset with more than 48 _,_ 000 data points. Given 14 attributes, e.g. age, sex, and occupation, the task is to determine whether a person earns over 50 _,_ 000 $ per year. Spambase [18] is a binary classification dataset containing 4 _,_ 601 data points. Given 57 numerical attributes, e.g. frequencies of words and symbols in a mail, the task is to determine whether that mail is spam. 

**Experimental Setup.** We tune our S-BDT and the state of the art by Maddock et al. [24] in a randomized grid-search that randomly selects without replacement hyperparameters of a grid search. In all experiments, we set _δ_ = 5 _·_ 10 _[−]_[8] . We use 5-fold cross-validation to obtain training and test datasets. In the hyperparameter search every setting is evaluated for 20 runs and the best-performing hyperparameter setting, again for 200 runs (for Spambase and _ε ≤_ 0 _._ 1: 1000 runs). Tests were run with an Intel Xeon Platinum 8168 2.7 GHz CPU and 32GB of RAM. The code of S-BDT is available at https://github.com/kirschte/sbdt. 

**Model performance.** We measure model performance on the test dataset by the root mean squared error (RMSE) for regression and AUC-ROC for classification. We also report the standard error of the mean, defined by the standard deviation of the runs divided by the square root of the number of runs [2]. The standard error represents the standard deviation of the distribution over the means of the runs, and as such gives an estimate on the precision of the sampled mean. The standard error decreases with an increasing number of runs, as the extent 

of chance variation is reduced. 

**Non-private baselines.** For the non-private baselines we implement the XGBoost [11] algorithm in our framework and denote _xgboost_ for reporting the utility when selecting the optimal, data-dependent split and _xgboost random splits_ when selecting splits uniformly at random from the split value range. 

**Learning a stream of non-IID data.** For the baseline, a naïve training approach for streams that adds extra training rounds for newly arrived data, and training with Rényi filter we investigate _T[′] ∈{T/_ 4 _, T/_ 2 _, T_ 3 _/_ 4 _, T }_ extra training rounds for _T_ regular training rounds. We rerun our RDP accountant for the extra training rounds, i.e. we adapt the noise scale of the extra round to the number of extra rounds. 

**Hyperparameters.** The evaluated hyperparameters are the number of trees _T_ regular _∈ {_ 5 _,_ 10 _,_ 25 _,_ 50 _,_ 100 _,_ 150 _,_ 200 _,_ 300 _,_ 400 _,_ 500 _,_ 600 _}_ , the depth of the trees _d ∈ {_ 2 _,_ 3 _,_ 5 _,_ 6 _}_ and the number of training rounds of iterative Hessian (cf. Sec. 4.1) _s ∈{_ 5 _,_ 30 _,_ 100 _}_ . We clip gradients with _g[∗] ∈ {_ 0 _._ 1 _,_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _,_ 0 _._ 9 _}_ and Hessians with _h[∗] ∈{_ 0 _._ 1 _,_ 0 _._ 25 _}_ . To simplify gradient clipping for regression we scale the regression labels to the range [ _−_ 1 _,_ 1]. We evaluate subsampling ratio _γ ∈{_ 0 _._ 005 _,_ 0 _._ 05 _,_ 0 _._ 1 _,_ 0 _._ 2 _}_ , leafbalanced noise parameter _r_ 1 _∈{_ 0 _._ 04 _,_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3 _, ...,_ 0 _._ 9 _}_ , ratio _ε_ init _/ε ∈{_ 10% _,_ 30% _}_ , clipping bound for initial score _m[∗] ∈{_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1 _._ 0 _}_ . We fix the privacy budget for DP releasing the dataset size for the initial score _ε_ ds = 0 _._ 005. We fix one regularization parameter _β_ = 2 and investigate _λ ∈{_ 1 _,_ 15 _}_ (Abalone and Spambase) and _λ ∈{_ 1 _,_ 10 _}_ (Adult). We investigate learning rate _η ∈{_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3 _}_ . 

To reduce the overhead for learning a stream of non-IID data, we restrict some hyperparameters to the range that was useful for IID data. For S-BDT we evaluate _T_ regular _∈ {_ 50 _,_ 100 _,_ 200 _,_ 400 _}_ for Abalone and _T_ regular _∈{_ 50 _,_ 200 _,_ 400 _}_ for Adult. We fix _ε_ init _/ε_ = 10%, _m[∗]_ = 1 _._ 0, _γ_ = 0 _._ 1, _η_ = 0 _._ 1 and _r_ 1 = 0 _._ 2 (Abalone) and _r_ 1 = 0 _._ 1 (Adult). For Maddock et al. we evaluate _T_ regular _∈{_ 5 _,_ 10 _,_ 25 _}_ for Abalone and _T_ regular _∈{_ 25 _,_ 50 _,_ 200 _}_ for Adult. We fix 5 rounds of training with iterative Hessian and _η_ = 0 _._ 3 For both, we evaluate _d ∈{_ 2 _,_ 4 _,_ 6 _}_ and _g[∗] ∈{_ 0 _._ 1 _,_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _}_ , _h[∗] ∈{_ 0 _._ 1 _,_ 0 _._ 25 _}_ . 

## **8.1 Comparing S-BDT to the SOTA [24]** 

We compare S-BDT to the best prior work [24] both including the following relevant algorithmic variations proposed by Maddock et al. [24]. For the split selection, we also use random splits which is the best-performing variant in Maddock et al. [24]. For random splits, we choose a feature uniformly at random and a split value uniformly at random from a pre-defined feature range. We evaluate two weight update methods, gradient boosting for regression and Newton boosting for classification. For regression, gradient and Newton boosting are the same, as we divide in the leaf by the number of samples in that leaf which is the same as the sum of Hessians. For generating split candidates we evaluate: No prior split candidates, i.e. sampling a split from the full pre-defined feature range uniformly at random, equidistant split candidates from the pre-defined feature range, and equidistant split candidates refined with iterative Hessian, an approach that refines the initial split candidates based on the aggregated Hessians of data points for each split candidate. The latter two approaches are proposed by Maddock et al. [24]. We evaluate two feature 

12 

**==> picture [231 x 510] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 4 Maddock et al.<br>S-BDT<br>3 . 2<br>xgboost (nonprivate)<br>3 . 0 xgboost random splits (nonprivate)<br>2 . 8 50% ε saving<br>2 . 6<br>better<br>2 . 4<br>2 . 2<br>2 . 0<br>0 . 1 0 . 2 0 . 3 0 . 5 1 . 0<br>ε (privacy budget)<br>(a) Dataset: Abalone (Regression)<br>1 . 00<br>better<br>0 . 95<br>0 . 90<br>0 . 85 30% ε saving<br>Maddock et al.<br>0 . 80<br>S-BDT<br>0 . 75 xgboost (nonprivate)<br>xgboost random splits (nonprivate)<br>0 . 70<br>0 . 010 0 . 025 0 . 050 0 . 100 0 . 500 1 . 000<br>ε (privacy budget)<br>(b) Dataset: Adult (Classification)<br>1 . 00<br>0 . 95<br>better<br>0 . 90<br>0 . 85<br>0 . 80 30% ε saving<br>0 . 75 Maddock et al.<br>0 . 70 S-BDT<br>xgboost (nonprivate)<br>0 . 65<br>xgboost random splits (nonprivate)<br>0 . 60<br>0 . 010 0 . 025 0 . 050 0 . 100 0 . 500 1 . 000<br>ε (privacy budget)<br>(RMSE)<br>error<br>regression<br>test<br>Mean<br>AUC<br>test<br>Mean<br>AUC<br>test<br>Mean<br>**----- End of picture text -----**<br>


(c) Dataset: **Spambase** (Classification) 

Figure 3: **Comparison of utility-privacy tradeoff of our S-BDT and the SOTA by Maddock et al. [24]** . Regression error (RMSE) (Abalone) and AUC (Adult and Spambase) of 200 runs (for Spambase and _ε ≤_ 0 _._ 1: 1000 runs) vs. privacy budget _ε_ (( _b_ ) and ( _c_ ) in log-scale). The transparent area is the standard error. 

interactions, cyclical, where each tree is trained on a single feature selected cyclically from all features, and random, where a random feature is selected in every split. 

We vary the privacy budget _ε_ and perform a hyperparameter search over the other parameters. Fig. 3 displays our findings and Tbl. 2a and Tbl. 2b the improvement of S-BDT over Maddock et al. [24] for a single privacy budget value. We observe that S-BDT achieves better per- 

|Technique|Mean test regression error (RMSE)|
|---|---|
|Maddock et al. [24]|2.939 _±_0_._019|
|+ Subsampling|2.782 _±_0_._008|
|+ DP initial score|2.760 _±_0_._009|
|+ Leaf-balanced noise|2.745 _±_0_._008|
|(a) Dataset:|**Abalone** (Regression)|
|Technique|Mean test AUC|
|Maddock et al. [24]|0.791 _±_0_._002|
|+ Subsampling|0.811 _±_0_._001|
|+ Leaf-balanced noise|0.825 _±_0_._001|



(b) Dataset: **Adult** (Classification) 

Table 2: **Ablation study** : Improvement of our S-BDT over the SOTA by Maddock et al. [24]. We report the RMSE (Abalone, _ε_ = 0 _._ 105) and the AUC (Adult, _ε_ = 0 _._ 02) each with a standard error of 200 runs. The parameters are chosen as in Fig. 4. 

formance for regression and classification than the SOTA by Maddock et al. For _ε_ = 0 _._ 25 on Abalone, the RMSE of S-BDT is 2.64 which Maddock et al. reach for _ε_ = 0 _._ 5, saving 50% in terms of epsilon (cf. Fig. 3a). For _ε_ = 0 _._ 053 on Adult, the AUC of S-BDT is 0.853 which Maddock et al. reach for _ε_ = 0 _._ 08, saving 30% in terms of epsilon (Fig. 3b). For _ε_ = 0 _._ 02 on Spambase, the AUC of S-BDT is 0.79 which Maddock et al. reach for _ε_ = 0 _._ 03, saving 30% in terms of epsilon (Fig. 3c). For the split selection, we observe that split refinement using Hessian information as proposed by Maddock et al. [24] is beneficial for their work but not for ours. 

S-BDT uses up to 600 training rounds, Maddock et al. up to 300. S-BDT takes 3.2 seconds to train 1000 trees with tree depth 6 on Abalone, and 8.2 seconds on Adult. With individual Rényi filter, S-BDT takes 5.1 seconds on Abalone and 25.7 seconds on Adult. We update the individual privacy budget by slightly rounding up the individual sensitivities _gi, hi_ , computing the individual privacy loss as an upper bound on the exact one and storing it for later use. This saves time cost and is proposed in prior work [33]. 

## **8.2 Random splits vs. data-dependent splits** 

Our evaluation supports the results of prior work [7, 24] that random splits compared to data-dependent splits cost some utility, yet with a limited effect. For random splits, we select uniformly at random a feature and uniformly at random a value from a pre-defined feature range. For data-dependent splits, we evaluate the best split across all features and possible feature values, i.e. data points for this feature, using a variant of the MSE gain as used in xgboost [11]. In fact, on the three datasets Abalone, Adult, and Spambase we observe in the non-private setting a gap of 0 _._ 07 in RMSE (Abalone), 0 _._ 008 in AUC (Adult), and 0 _._ 004 in AUC (Spambase) between data-independent random splits and data-dependent gain-based splits (cf. Fig. 3). This demonstrates that the privacy leakage of tree learners can be reduced to the leaves and that random splits can be of interest even in a non-private setting due to their significantly faster computation time. 

13 

**==> picture [484 x 260] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 10<br>3 . 10 3 . 10 Maddock et al.<br>3 . 05 MaddockS-BDT et al. 3 . 05 MaddockS-BDT et al. 33 .. 0500 S-BDT<br>3 . 00 3 . 00<br>2 . 95 better<br>2 . 95 2 . 95 better<br>2 . 90<br>2 . 90 better 2 . 90<br>2 . 85<br>2 . 85 2 . 85<br>2 . 80<br>2 . 80 2 . 80<br>2 . 75<br>2 . 75 2 . 75<br>2 . 70<br>2 . 70 2 . 70 0 . 01 0 . 10 0 . 20 0 . 30 0 . 40 0 . 50<br>0 . 1 0 . 2 0 . 3 0 . 4 0 . 5 0 . 6 0 . 7 0 . 8 0 . 9 0 . 01 0 . 20 0 . 40 0 . 60 0 . 80 1 . 00 ε init /ε (privacy budget ratio for initial score)<br>r 1 (leaf-balanced noise parameter) γ (subsampling ratio)<br>(a) Dataset: Abalone. Varying r 1 (b) Dataset: Abalone. Varying γ (c) Dataset: Abalone. Varying<br>(leaf-balanced noise parameter) (subsampling ratio) (privacy budget ε ratioinit /ε for initial score)<br>0 . 84 Maddock et al. 0 . 84 Maddock et al. 0 . 84 Maddock et al.<br>S-BDT S-BDT S-BDT<br>0 . 82 0 . 82 0 . 82<br>0 . 80 0 . 80 0 . 80<br>0 . 78 better 0 . 78 better 0 . 78 better<br>0 . 76 0 . 76 0 . 76<br>0 . 1 0 . 2 0 . 3 0 . 4 0 . 5 0 . 6 0 . 7 0 . 8 0 . 9 0 . 005 0 . 200 0 . 400 0 . 600 0 . 800 1 . 000 0 . 01 0 . 10 0 . 20 0 . 30 0 . 40 0 . 50<br>r 1 (leaf-balanced noise parameter) γ (subsampling ratio) ε init /ε ) (privacy budget ratio for initial score)<br>(d) Dataset: Adult. Varying r 1 (e) Dataset: Adult. Varying γ (f) Dataset: Adult. Varying ε init /ε<br>(leaf-balanced noise parameter) (subsampling ratio) (privacy budget ratio for initial score)<br>(RMSE)<br>(RMSE) (RMSE)<br>error<br>error error<br>regression<br>regression regression<br>test<br>test test<br>Mean<br>Mean Mean<br>AUC AUC AUC<br>test test test<br>Mean Mean Mean<br>**----- End of picture text -----**<br>


Figure 4: **Ablation studies of our improvements:** Regression error (RMSE) and AUC of 200 runs. The transparent area is the standard error. For Abalone we set _ε_ = 0 _._ 105, number of trees _T_ regular = 150, depth _d_ = 2, subsampling ratio _γ_ = 0 _._ 1, leaf-balanced noise parameter _r_ 1 = 0 _._ 2, privacy budget ratio for initial score _ε_ init _/ε_ = 0 _._ 1, _ε_ ds = 0 _._ 005 and clipping bound _g[∗]_ = 0 _._ 1. For Adult, we set _ε_ = 0 _._ 02, _T_ regular = 200, _d_ = 5, _γ_ = 0 _._ 005, _r_ 1 = 0 _._ 1, _g[∗]_ = 0 _._ 5 _, h[∗]_ = 0 _._ 1. A smaller _r_ 1 value means more privacy budget for Hessian sum compared to gradient sum. _r_ 1 = 0 _._ 5 deactivates leaf-balanced noise, _γ_ = 1 _._ 0 subsampling and _ε_ init _/ε_ = 0 _._ 0 the initial score. 

**==> picture [185 x 286] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 3<br>Maddock et al. (naive extra rounds)<br>3 . 2<br>S-BDT (naive extra rounds)<br>3 . 1 S-BDT (Rényi flter extra rounds)<br>3 . 0<br>2 . 9<br>2 . 8<br>2 . 7<br>2 . 6 better<br>2 . 5<br>0 . 05 0 . 10 0 . 20 0 . 30 0 . 50<br>ε (privacy budget)<br>(a) Dataset: Abalone (Regression)<br>1 . 00<br>S-BDT (Rényi filter extra rounds)<br>0 . 95 S-BDT (naive extra rounds)<br>0 . 90 Maddock et al. (naive extra rounds)<br>0 . 85 better<br>0 . 80<br>0 . 75<br>0 . 70<br>0 . 65<br>0 . 01 0 . 02 0 . 03 0 . 10 0 . 50<br>ε (privacy budget) log-scaled<br>(b) Dataset: Adult (Classification)<br>(RMSE)<br>error<br>regression<br>test<br>Mean<br>AUC<br>test<br>Mean<br>**----- End of picture text -----**<br>


Figure 5: **Learning a stream of non-IID data:** Regression error (RMSE) (Abalone) and AUC (Adult) of 200 runs vs. privacy budget _ε_ . The transparent area is the standard error. 

## **8.3 Impact of algorithmic parameters** _r_ 1 _, γ, ε_ **init** _/ε_ 

We explore the impact of parameters for our leaf-balanced noise, subsampling, and our initial score on model performance. We fix the optimal setting for our S-BDT and then vary a single one of the parameters. Our findings are displayed in Fig. 4. For leaf-balanced noise, choosing the range 0 _._ 05 _≤ r_ 1 _≤_ 0 _._ 2 is most helpful which assigns more budget to the gradient sum (leaf numerator) (cf. Fig. 4a, Fig. 4d). Choosing a small subsampling ratio _γ_ is generally beneficial ( _γ <_ 0 _._ 3) but if the chosen _γ_ is too small, the privacy amplification cannot outweigh the small amount of data points used for a single tree which can hurt model performance (cf. Fig. 4e, Fig. 4b). We fix _ε_ ds = 0 _._ 005 and tune _ε_ init. Spending privacy budget _ε_ init on the initial score improves performance on Abalone (cf. Fig. 4c), but choosing a fraction _ε_ init _/ε >_ 0 _._ 3 leaves too little privacy budget for the trees and hurts model performance. 

## **8.4 Learning a stream of non-IID data** 

We study the impact of an individual Rényi filter for learning a stream of non-IID data. On the stream, all data points with a regression label above the mean of the data set arrive after the regular training rounds are over. For classification, all data points with label 1 arrive late. As the baseline, we adopt a naïve training approach for streams by adding extra training rounds that only train with the newly arrived data. Our S-BDT with an individual Rényi filter instead can incorporate not only newly arrived data but also formerly seen data points that still have a privacy budget left. We vary the privacy budget _ε_ and perform 

14 

a hyperparameter search over the other parameters. For the SOTA by Maddock et al. [24] we adopt the naïve training approach for streams. Fig. 5 displays our findings: S-BDT with individual Rényi filter performs better than the naïve baseline. This benefit comes without spending any additional privacy budget, an important feature of an individual Rényi filter. 

## **9 Other Related work** 

In the literature, there are two lines of secure (distributed) GBDT training: Cryptography-based [20, 19] and DPbased methods [7, 23, 24, 28, 32, 15]. Cryptography-based methods are parallel to our approach as they focus the attack surface on the computation of GBDTs and not on the release. They frequently utilize secure multiparty computation (MPC) or homomorphic encryption (HE). For DP-based GBDTs, Maddock et al. [24] is the closest to us as they outperform prior works DP-EBM [28], FEVERLESS [32], and DP-RF [15]. Other work [7] focuses on random forests. Another line of work [23] provides suboptimal leaf noising in _O_ ( _g[∗]_ ) and not in _O_ ( _[g][∗] /n_ ) with _g[∗]_ as the gradient clipping bound and _n_ the number of data points in a leaf together with data-dependent gain-based DP splits which is already covered in Maddock et al. [24]. 

## **10 Conclusion** 

We introduced S-BDT for learning GBDTs, which displays strong utility-privacy performance for the Abalone, Adult, and Spambase datasets. Compared to prior work, S-BDT incorporates three techniques. To reduce the degree of noise used during training and increase the number of trees, we incorporate subsampling and leaf-balanced noise and prove tight privacy bounds. We adopt so-called individual Rényi filters to ensure that data points that were used in prior trees but were underutilized can be used for training the next trees. As a result, S-BDT is well-suited for training models on streams of non-IID data where the intermediary models are released. 

We present a combination of tight individual Rényi DP bound for non-spherical multivariate Gaussian mechanism with leaf-balanced noise and a novel generalized condition for tight privacy bounds for subsampling. This combination might be of independent interest, as it enables a better calibration of the noise for functions with an arbitrary number of output dimensions. The novel condition solely requires the ( _α, ρ_ ( _α_ ))-Rényi DP bound of the subsampled mechanism to be linear in _α_ . 

_Neural Information Processing Systems_ , Vol. 31. Curran Associates, Inc. 

- [4] Borja Balle, Gilles Barthe, Marco Gaboardi, Justin Hsu, and Tetsuya Sato. 2020. Hypothesis Testing Interpretations and Renyi Differential Privacy. In _Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics (Proceedings of Machine Learning Research, Vol. 108)_ . PMLR, 2496– 2506. 

- [5] Barry Becker and Ronny Kohavi. 1996. Adult. UCI Machine Learning Repository. `https://archive. ics.uci.edu/dataset/2/adult` 

- [6] James Henry Bell, Kallista A. Bonawitz, Adrià Gascón, Tancrède Lepoint, and Mariana Raykova. 2020. Secure Single-Server Aggregation with (Poly)Logarithmic Overhead. In _Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security_ . Association for Computing Machinery, 1253–1269. 

- [7] Mariusz Bojarski, Anna Choromanska, Krzysztof Choromanski, and Yann LeCun. 2014. Differentiallyand non-differentially-private random decision trees. _preprint arXiv:1410.6973_ (2014). 

- [8] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. 2017. Practical Secure Aggregation for Privacy-Preserving Machine Learning. In _Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security_ . Association for Computing Machinery, 1175–1191. 

- [9] Clément L Canonne, Gautam Kamath, and Thomas Steinke. 2020. The Discrete Gaussian for Differential Privacy. In _Advances in Neural Information Processing Systems_ , Vol. 33. Curran Associates, Inc., 15676– 15688. 

- [10] Thee Chanyaswad, Alex Dytso, H Vincent Poor, and Prateek Mittal. 2018. Mvg mechanism: Differential privacy under matrix-valued query. In _Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security_ . 230–246. 

- [11] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’16)_ . Association for Computing Machinery, 785–794. 

## **References** 

- [1] Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016. Deep Learning with Differential Privacy. In _Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security_ . ACM. 

- [2] Douglas G Altman and J Martin Bland. 2005. Standard deviations and standard errors. _BMJ_ 331, 7521 (2005), 903. 

- [3] Borja Balle, Gilles Barthe, and Marco Gaboardi. 2018. Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences. In _Advances in_ 

- [12] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam D. Smith. 2006. Calibrating Noise to Sensitivity in Private Data Analysis. In _Theory of Cryptography, TCC 2006 (Lecture Notes in Computer Science, Vol. 3876)_ . Springer, Berlin, Heidelberg, 265–284. 

- [13] Cynthia Dwork and Aaron Roth. 2014. The Algorithmic Foundations of Differential Privacy. _Found. Trends Theor. Comput. Sci._ 9, 3-4 (2014), 211–407. 

- [14] Vitaly Feldman and Tijana Zrnic. 2021. Individual Privacy Accounting via a Rényi Filter. In _Advances in Neural Information Processing Systems_ , Vol. 34. Curran Associates, Inc., 28080–28091. 

15 

- [15] Sam Fletcher and Md Zahidul Islam. 2017. Differentially private random decision forests using smooth sensitivity. _Expert Systems with Applications_ 78 (2017), 16–31. 

- [16] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. 2015. Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. In _Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS ’15)_ . Association for Computing Machinery, New York, NY, USA, 1322–1333. 

- [17] Jerome H Friedman. 2001. Greedy function approximation: a gradient boosting machine. _Annals of statistics_ (2001), 1189–1232. 

- [18] Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt. 1999. Spambase. UCI Machine Learning Repository. `https://archive.ics.uci. edu/dataset/94/spambase` 

- [19] Yufan Jiang, Fei Mei, Tianxiang Dai, and Yong Li. 2024. SiGBDT: Large-Scale Gradient Boosting Decision Tree Training via Function Secret Sharing. In _Proceedings of the 19th ACM Asia Conference on Computer and Communications Security_ (Singapore, Singapore) _(ASIA CCS ’24)_ . Association for Computing Machinery, New York, NY, USA, 274–288. 

- [20] Wen jie Lu, Zhicong Huang, Qizhi Zhang, Yuchen Wang, and Cheng Hong. 2023. Squirrel: A Scalable Secure Two-Party Computation Framework for Training Gradient Boosting Decision Tree. In _32nd USENIX Security Symposium (USENIX Security 23)_ . USENIX Association, Anaheim, CA, 6435–6451. 

- [21] Antti Koskela, Marlon Tobaben, and Antti Honkela. 2023. Individual Privacy Accounting with Gaussian Differential Privacy. In _The Eleventh International Conference on Learning Representations, ICLR_ . 

- [22] Ninghui Li, Wahbeh Qardaji, and Dong Su. 2012. On sampling, anonymization, and differential privacy or, k-anonymization meets differential privacy. In _Proceedings of the 7th ACM Symposium on Information, Computer and Communications Security (ASIACCS ’12)_ . Association for Computing Machinery, New York, NY, USA, 32–33. 

   - [27] Warwick Nash, Tracy Sellers, Simon Talbot, Andrew Cawthorn, and Wes Ford. 1995. Abalone. UCI Machine Learning Repository. `https://archive.ics. uci.edu/dataset/1/abalone` 

   - [28] Harsha Nori, Rich Caruana, Zhiqi Bu, Judy Hanwen Shen, and Janardhan Kulkarni. 2021. Accuracy, Interpretability, and Differential Privacy via Explainable Boosting. In _Proceedings of the 38th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 139)_ . PMLR, 8227– 8237. 

   - [29] Alfréd Rényi. 1961. On measures of entropy and information. In _Proceedings of the fourth Berkeley symposium on mathematical statistics and probability, volume 1: contributions to the theory of statistics_ , Vol. 4. University of California Press, 547–562. 

   - [30] César Sabater, Florian Hahn, Andreas Peter, and Jan Ramon. 2022. Private Sampling with Identifiable Cheaters. _Proceedings on Privacy Enhancing Technologies_ 2023 (01 2022). 

   - [31] David M Sommer, Sebastian Meiser, and Esfandiar Mohammadi. 2019. Privacy Loss Classes: The Central Limit Theorem in Differential Privacy. _Proceedings on Privacy Enhancing Technologies_ 2 (2019), 245–269. 

   - [32] Rui Wang, Oğuzhan Ersoy, Hangyu Zhu, Yaochu Jin, and Kaitai Liang. 2022. FEVERLESS: Fast and Secure Vertical Federated Learning based on XGBoost for Decentralized Labels. _IEEE Transactions on Big Data_ (2022), 1–15. 

   - [33] Da Yu, Gautam Kamath, Janardhan Kulkarni, TieYan Liu, Jian Yin, and Huishuai Zhang. 2023. Individual Privacy Accounting for Differentially Private Stochastic Gradient Descent. _Transactions on Machine Learning Research_ (2023). 

   - [34] Yuqing Zhu and Yu-Xiang Wang. 2019. Poisson Subsampled Rényi Differential Privacy. In _Proceedings of the 36th International Conference on Machine Learning_ , Vol. 97. PMLR, 7634–7642. 

- [23] Qinbin Li, Zhaomin Wu, Zeyi Wen, and Bingsheng He. 2020. Privacy-Preserving Gradient Boosting Decision Trees. _Proceedings of the AAAI Conference on Artificial Intelligence_ 34, 01 (2020), 784–791. 

- [24] Samuel Maddock, Graham Cormode, Tianhao Wang, Carsten Maple, and Somesh Jha. 2022. Federated Boosted Decision Trees with Differential Privacy. In _Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security_ . ACM. 

- [25] H. Brendan McMahan, Daniel Ramage, Kunal Talwar, and Li Zhang. 2018. Learning Differentially Private Recurrent Language Models. In _International Conference on Learning Representations_ . 

- [26] Ilya Mironov. 2017. Rényi Differential Privacy. In _2017 IEEE 30th Computer Security Foundations Symposium (CSF)_ . 263–275. 

16 

## **Appendix** 

## **A Further experiments** 

## **Individual Rényi filter for regular training.** 

In Sec. 8.4 we evaluate the individual Rényi filter for learning a stream of non-IID data. Here, we investigate the impact of an individual Rényi filter for regular training. 

Prior works [14][21] have empirically shown that an individual Rényi filter can improve performance when choosing hyperparameters such as a clipping bound suboptimally. Our findings are displayed in Fig. 6. We can also observe the effect of improvement (cf. Fig. 6b) when we increase the gradient clipping bound from its optimal value to twice the value. We investigate the number of extra training rounds _T_ extra _∈{T/_ 4 _, T/_ 2 _, T_ 3 _/_ 4 _, T }_ . 

|Setting||Time cost (seconds)|
|---|---|---|
|S-BDT|no Rényi flter|3.2|
|S-BDT|with Rényi flter|5.1|
||(a) **Abalone**||
|Setting||Time cost (seconds)|
|S-BDT|no Rényi flter|8.2|
|S-BDT|with Rényi flter|25.7|
||(b) **Adult**||



Table 3: **Time cost** of S-BDT on Abalone and Adult. We train _T_ regular = 1000 rounds with tree depth 6. For the individual Rényi filter we add _T_ extra = 1000 extra rounds. 

**Time cost.** 

We investigate the time cost of S-BDT. The individual Rényi filter is quite impactful here. An individual Rényi 

**==> picture [231 x 349] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 00<br>S-BDT (no Rényi filter)<br>2 . 95<br>S-BDT (with Rényi filter)<br>2 . 90 better<br>2 . 85<br>2 . 80<br>2 . 75<br>2 . 70<br>2 . 65<br>2 . 60<br>0 . 10 0 . 15 0 . 20<br>g [∗] (gradient clipping bound)<br>(a) Dataset: Abalone (Regression)<br>1 . 00<br>S-BDT (no Rényi filter)<br>0 . 95 S-BDT (with Rényi flter)<br>0 . 90<br>better<br>0 . 85<br>0 . 80<br>0 . 75<br>0 . 5 0 . 6 0 . 7 0 . 8 0 . 9 1 . 0<br>g [∗] (gradient clipping bound)<br>(b) Dataset: Adult (Classification)<br>(RMSE)<br>error<br>regression<br>test<br>Mean<br>AUC<br>test<br>Mean<br>**----- End of picture text -----**<br>


Figure 6: **Ablation study of an individual Rényi filter tailored to S-BDT** . Regression error (RMSE) (Abalone) and AUC (Adult) of 200 runs. The transparent area is the standard error. For Abalone we set _ε_ = 0 _._ 1, number of trees 150, depth 2, subsampling ratio _γ_ = 0 _._ 1, leaf-balanced noise parameter _r_ 1 = 0 _._ 2, privacy budget ratio for initial score _ε_ init _/ε_ = 0 _._ 1 and clipping bound _g[∗]_ = 0 _._ 1. For Adult we set _ε_ = 0 _._ 02, number of trees 200, depth 5, subsampling ratio _γ_ = 0 _._ 005, leaf-balanced noise parameter _r_ 1 = 0 _._ 1, privacy budget ratio for initial score _ε_ init _/ε_ = 0 _._ 1 and clipping bounds _g[∗]_ = 0 _._ 5 _, h[∗]_ = 0 _._ 1. 

filter demands updating the individual privacy budget for every data point in every iteration and extra training rounds after the regular training rounds, both increase the time costs. We update the individual privacy budget by slightly rounding up the individual sensitivities _gi, hi_ and then computing the individual privacy loss as an upper bound on the exact individual privacy loss. This strategy saves time cost when the computed individual privacy losses are stored for later use and is proposed in prior work [33]. We investigate training for 1000 rounds and a tree depth 6 and compare no individual privacy accounting and training with an individual Rényi filter for _T_ extra = 1000 extra rounds. We observe the time cost displayed in Tbl. 3. 

## **B Postponed proofs** 

## **B.1 Useful RDP Properties** 

**Lemma 24** (From RDP to individual RDP) **.** _Let M be any mechanism satisfying_ ( _α, ρ_ ( _α_ )) _-Rényi differential privacy. Then M satisfies_ ( _α, ρ_ ( _α_ )) _-individual Rényi differential privacy for data point x._ 

_Proof._ Let _M_ be as defined in the lemma’s statement. Let _X, X[′]_ be two neighboring datasets, then 

**==> picture [110 x 11] intentionally omitted <==**

Now, let _D ∼di D[′]_ be two neighboring datasets differing in _di_ . Since _ρ_ ( _α_ ) is a upper bound on the Rényi divergence of _M_ ( _X_ ) _, M_ ( _X[′]_ ) for arbitrary inputs, it also applies for the specific inputs _D, D[′]_ : 

**==> picture [109 x 11] intentionally omitted <==**

Thus, _M_ is also ( _α, ρ_ ( _α_ )) individual RDP. 

**Corollary 25** (Adaptive sequential Composition for individual RDP) **.** _Let M be a sequence of adaptively chosen mechanisms Mi_ : Π _[i] j[−]_ =1[1] _[R][j][× X][�→R][i][(][i]_[ = 1] _[,]_[ 2] _[, ..., k]_[)] _[,][each] providing_ ( _α, ρi_ ) _-individual Rényi differential privacy for data point x. Then for any α M is_ ( _α,_[�] _[k] i[ρ][i]_[(] _[α]_[))] _[-individual] Rényi differentially private for data point x._ 

_Proof._ This corollary follows directly from Thm. 5. The proof of this theorem is parametric in a set of neighboring datasets. By choosing the set of neighboring datasets as all ( _X, X ∪{ x }_ ), the statement follows. 

17 

## **B.2 Main Theorem** 

We state the full Thm. 18 and the full proof. 

## **B.3 RDP Proofs of each S-BDT Component** 

## **B.3.1 DPInitScore** 

**Theorem 23** (Main theorem) **.** _Alg. 2 (_ _`TrainSBDT` ) is_ ( _α, ρ_ ( _α_ ) _)-Rényi DP._ 

We recall Thm. 18 and state the full proof. 

**Theorem 18.** _Alg. 3 (_ _`DPInitialScore` ) with clipping bound m[∗] , DP approximated dataset size |D|priv and f_ ( _ε_ ) = log � 2 _αα−_ 1[exp] �( _α −_ 1) _· ε_ � + 2 _αα−−_ 11[exp] � _− α · ε_ �� _satisfies_ � _α, α−_ 1 1[(] _[f]_[(] _[ε][init]_[) +] _[ f]_[(] _[ε][ds]_[))] � _-RDP._ 

_Proof._ We prove an ( _α, ρ_ ( _α_ ))-RDP bound for the learning algorithm `TrainSBDT` . 

Let _M_ 0 := `DPInitScore` , _Mi_ := `TrainSingleTree` _◦_ `Filter` _[ρ] α_[(] _[α]_[)] _∀i >_ 0. `Filter` _[ρ] α_[(] _[α]_[)] denotes the individual Rényi filter part of our algorithm (lines 6 to 8 in Alg. 2): The filter computes individual RDP privacy losses for `TrainSingleTree` for every data point and then filters out data points that have exceeded the individual RDP bound of _ρ_ ( _α_ ). 

> _Proof._ Let _X ∼ X[′]_ be two neighboring datasets. Assume without loss of generality that _X_ and _X[′]_ differ in data point ( _x[′] , y[′]_ ). We first analyze the sensitivities of `DPInitialScore` ’s subfunctions. 

Let _g_ ds denote the subfunction of `DPInitialScore` without noise, that computes the dataset size. The sensitivity of _g_ ds is ∆ _g_ ds = 1 because we investigate unbounded DP, and adding or removing a single element from the dataset can change its size by at most 1. 

Let _T_[(0] _[..k]_[)] ( _X_ ) := _Mk_ ( _X, Mk−_ 1( _X, ..._ ( _X, M_ 0( _X_ )))) be the nested applications of mechanism _Mi_ , comprising the learning algorithm `TrainSBDT` , where _T_[(] _[v..w]_[)] ( _X_ ) := _Mw_ ( _X, Mk−_ 1( _X, ..._ ( _X, Mv_ ( _X_ )))). The output of the training algorithm `TrainSBDT` , which we call an observation, is a sequence of trees _o_ 0 _, ..., ok_ , consisting of an initial score _o_ 0 and trees _oi_ output by the application of _Mi_ . We write **o** := _o_[(0] _[..k]_[)] = ( _o_ 0 _, o_ 1 _, ..., ok_ ), where _o_[(] _[v..w]_[)] = ( _ov, ..., ow_ ). 

**==> picture [234 x 146] intentionally omitted <==**

**==> picture [144 x 10] intentionally omitted <==**

**==> picture [191 x 35] intentionally omitted <==**

Applying the RDP sequential composition bound of Thm. 5 we get 

**==> picture [465 x 69] intentionally omitted <==**

**==> picture [186 x 53] intentionally omitted <==**

Let _f_ ( _ε_ ) = log � 2 _αα−_ 1[exp] �( _α−_ 1) _·ε_ �+ 2 _[α] α[−] −_[1] 1[exp] � _−α·ε_ ��, applying the RDP bound for `DPInitScore` of Thm. 18 we get 

We split the integral into three parts 

**==> picture [191 x 37] intentionally omitted <==**

**==> picture [188 x 214] intentionally omitted <==**

By definition of _T_[(1] _[..k]_[)] , _T_[(1] _[..k]_[)] is a sequence of _Mi_ ( _i_ = 1 _,_ 2 _, ..., k_ ) where each _Mi_ consists of two operations: (1) The filter operation `Filter` _[ρ] α_[(] _[α]_[)] that applies the privacy filter and filters out data points that have exceeded the individual RDP bound _ρ_ ( _α_ ). (2) The mechanism `TrainSingleTree` that trains a tree on the filtered dataset. 

In Thm. 19 we bound the individual RDP privacy losses for `TrainSingleTree` , so our filter is correct, i.e. it will always filter out data points that have exceeded the individual RDP bound _ρ_ ( _α_ ). By Thm. 12 the sequence of mechanisms _T_[(1] _[..k]_[)] then satisfies the following RDP bound 

We use this result to define a function: 

**==> picture [131 x 18] intentionally omitted <==**

18 

We show, that `DPInitialScore` satisfies RDP. The output of `DPInitialScore` , which we call an observation, is a post-processed version _[o]_[sum] _/o_ ds of a pair of observations ( _o_ ds _, o_ sum) consisting of the dataset size and the sum of clipped labels. As RDP is preserved by post-processing (cf. Thm. 9) we investigate the pair of observations output by `DPInitialScore` prior to post-processing. Let _M_ ds _, M_ sum be the two mechanisms computing the noisy dataset size and the noisy sum of clipped labels. 

## _Dα_ ( `DPInitialScore` ( _X_ ) _||_ `DPInitialScore` ( _X[′]_ )) 

**==> picture [226 x 14] intentionally omitted <==**

Applying the RDP sequential composition bound of Thm. 5 we get 

**==> picture [218 x 46] intentionally omitted <==**

_M_ ds _, M_ sum are Laplace mechanisms with noise scales ∆ _g_ ds _/ε_ ds, ∆ _g_ sum _/ε_ init and sensitivities ∆ _g_ ds , ∆ _g_ sum so we can apply the previously obtained function _f_ ( _ε_ ): 

**==> picture [102 x 17] intentionally omitted <==**

## **B.3.2 DPLeaf** 

We state the full Cor. 17. 

**Corollary 17** (RDP of Gaussian Mechanism with leaf-balanced non-spherical noise) **.** _Let f_ : _X �→_ R _[D] denote a function from an arbitrary input X ∈X to a set of scalars. Let the function f projected to its d-th output have bounded sensitivity sd ∈_ R _. Let Y ∼N_ ( **0** _,_ Σ) _be a random variable of non-spherical multivariate Gaussian noise with covariance matrix_ Σ := _D[−]_[1] _·_ diag( _r_ 1 _[−]_[1] _[s]_ 1[2] _[σ]_[2] _[, . . . , r] D[−]_[1] _[s] D_[2] _[σ]_[2][)] _[where][a] given variance σ_[2] _∈_ R[+] _is weighted in each dimension d individually by s_[2] _d[and][r][d][∈]_[R][+] _[such][that]_[�] _[D] d_ =1 _[r][d]_[= 1] _[.][Then] the Gaussian mechanism M_ ( _X_ ) _�→{ f_ ( _X_ ) _d_ + _Yd }[D] d_ =1 _[is] D_ ( _α, ρ_ ( _α_ )) _-RDP with ρ_ ( _α_ ) = _α ·_ 2 _σ_[2] _[.]_ 

_Proof._ The corollary follows directly from Thm. 16 for a challenge data point with a worst-case sensitivity: _sd_ . 

We recall Cor. 21 and state the full proof. 

**Corollary 21.** _Alg. 5 (_ _`DPLeaf` ) satisfies_ ( _α, α ·_ 2 _σ_ 2 _leaf_[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2][+] _[r]_[2] ( _g[·][|][∗][g]_ ) _[i]_[2] _[|]_[2] �) _-individual RDP for data point xi with gradient gi and Hessian hi and with σleaf_[2] _[as][the][unweighted] variance of the leaf Gaussian._ 

_Proof._ For arbitrary input _X_ , `DPLeaf` without noise computes a function _f_ ( _X_ ) _�→_ �� _xi∈X[h][i][,]_[ �] _xi∈X[g][i]_ � where _gi, hi_ are the gradient and Hessian of data point _xi ∈ X_ . The first output of _f_ is _|hi|_ -sensitivity bounded with respect to _X ∼xi X[′]_ (i.e., _X_ and _X[′]_ only differ in _xi_ ), the second output of _f_ is _|gi|_ -sensitivity bounded with respect to _X ∼xi X[′]_ , as _X_ and _X[′]_ only differ in _xi_ : 

**==> picture [164 x 30] intentionally omitted <==**

In lines 3 and 5, `DPLeaf` applies leaf-balanced non-spherical bivariate Gaussian noise _Y ∼N_ ( **0** _,_ Σ) with covariance matrix Σ :=[1] 2[diag][(] _[r]_ 1 _[−]_[1][(] _[h][∗]_[)][2] _[σ]_ leaf[2] _[, r]_ 2 _[−]_[1][(] _[g][∗]_[)][2] _[σ]_ leaf[2][)][ to the output] of _f_ , denoted as _M_ ( _X_ ) _�→{f_ ( _X_ )1 + _Y_ 1 _, f_ ( _X_ )2 + _Y_ 2 _}_ . By Thm. 16, _M_ satisfies ( _α, ρ_ ( _α_ ))-individual RDP for _xi_ with _ρ_ ( _α_ ) = _α ·_ 2 _σ_ 2leaf[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2] + _[r]_[2] ( _g[·][|][∗][g]_ ) _[i]_[2] _[|]_[2] �. `DPLeaf` finally combines the outputs _f_ ( _X_ )1 _, f_ ( _X_ )2 of _f_ in line 6 via post-processing (cf. Thm. 9) without additional privacy leakage. 

We state the full Cor. 22. 

**Corollary 22.** _Alg. 5 (_ _`DPLeaf` ) satisfies_ ( _α,[α] /σleaf_[2][)] _[-RDP] with σleaf_[2] _[as][the][unweighted][variance][of][the][leaf][Gaussian.]_ 

_Proof._ The corollary follows directly from Cor. 21 for a challenge data point with worst-case sensitivities _g[∗] , h[∗]_ . 

## **B.3.3 TrainSingleTree** 

_Remark_ 26 _._ Privacy amplification by subsampling (Thm. 14) can be applied to individual RDP as well. The proof of Thm. 14 assumes a bound on the worst-case individual RDP of some mechanism _M_ and computes the worst-case individual RDP of the subsampled variant _M[P]_ of _M_ . Now, if we only assume an individual RDP bound for _M_ for some data point _xi_ we can analogously utilize Thm. 14 to obtain an individual RDP bound of subsampled _M[P]_ for _xi_ . We recall Thm. 19 and state the full proof. **Theorem 19.** _Let r_ 1 _, r_ 2 _be defined as in Thm. 16. Let σleaf_[2] _[be][the][unweighted][variance][of][the][leaf][Gaussian.][Let] aγ_ : N _×_ R _�→_ R _denote the privacy amplification of Thm. 14 with subsampling ratio γ. Then, for data point xi with gradient gi and Hessian hi, Alg. 4 (_ _`TrainSingleTree` ) satisfies_ ( _α, aγ_ ( _α, α ·_ 2 _σ_ 2 _leaf_[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2][+] _[r]_[2] ( _g[·][|][∗][g]_ ) _[i]_[2] _[|]_[2] �)) _-individual RDP._ 

_Proof._ Assume that _M_ is the mechanism computing the output of Alg. 4, when no subsampling is applied. Then _M_ comprises a sequence of two mechanisms _MR_ computing the random tree (line 2 in Alg. 4) and _ML_ computing a leaf value (line 4 in Alg. 4) for each leaf. Let _MA_ be the mechanism that outputs all _k_ of the leaves’ values at once, i.e. _MA_ ( _X_ ) = ( _ML_ ( _X_ ) _, ML_ ( _X_ ) _, ..., ML_ ( _X_ )). 

Let _o_ := _o_[(] _[R,]_[1] _[..k]_[)] = ( _oR, o_ 1 _, o_ 2 _, ..., ok_ ) be some observation of _M_ , comprising the output random tree and _k_ values for _k_ leaves of the random tree. Define _o_[(1] _[..k]_[)] = ( _o_ 1 _, o_ 2 _, ..., ok_ ), let _X, X[′]_ be two neighboring datasets. 

**==> picture [219 x 33] intentionally omitted <==**

We apply the individual RDP sequential composition bound of Cor. 25 and get 

**==> picture [195 x 35] intentionally omitted <==**

Since the splits are data-independent we have 

**==> picture [163 x 15] intentionally omitted <==**

19 

As the splits are chosen uniformly at random and due to the law of total probability, it suffices to consider an arbitrary but fixed splitting function _s_ with _s_ ( _X_ ) = ( _Xi_ ) _i[k]_ =1[that] partitions the data into _k_ partitions for _k_ leaves. 

**==> picture [205 x 35] intentionally omitted <==**

Each splitting function _s_ data-independently partitions the dataset into _k_ distinct subsets; hence, for neighboring datasets _D ∼xi D[′]_ in unbounded DP we get _s_ ( _D_ ) = ( _Xj_ ) _[k] j_ =1[and] _[s]_[(] _[D][′]_[) = (] _[X] j[′]_[)] _[k] j_ =1[such][that][for][one] _[i][X][i]_[and] _Xi[′]_[differs][in][the][one][element] _[x][i]_[(arbitrary][but][fixed)][and] for all _j_ = _i Xj_ = _Xj[′]_[.] 

**==> picture [222 x 53] intentionally omitted <==**

We apply the individual RDP bound of Cor. 21: 

**==> picture [129 x 18] intentionally omitted <==**

Finally, we show that `TrainSingleTree` with subsampling applied, yields a privacy amplification. As stated in Remark 26, we can utilize the privacy amplification by subsampling (Thm. 14) for individual RDP. The individual RDP bound of `TrainSingleTree` is _α ·_ 2 _σ_ 2leaf[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2] + _r_ 2( _g·|[∗] g_ ) _i_[2] _|_[2] � and is linear in _α_ . We use our condition for tight subsampling bounds (Lem. 15) to follow that the PearsonVajda pseudo-divergence of _DX[α]_ ( _M_ ( _X_ ) _||M_ ( _X[′]_ )) _≥_ 0 for all odd _α ≥_ 1. This allows us to use the tight subsampling bound of Thm. 14 to obtain _aγ_ ( _α, α ·_ 2 _σ_ 2leaf[2] _[·]_ � _r_ 1( _h·|[∗] h_ ) _i_[2] _|_[2] + 

_r_ 2( _g·|[∗] g_ ) _i_[2] _|_[2] �) as individual RDP bound. 

We state the full Cor. 20. 

**Corollary 20.** _Let σleaf_[2] _[be][the][unweighted][variance][of] the leaf Gaussian. Let aγ_ : N _×_ R _�→_ R _denote the privacy amplification of Thm. 14 with subsampling ratio γ. Then_ _`TrainSingleTree` (cf. Alg. 4) is_ ( _α, aγ_ ( _α,[α] /σleaf_[2][))] _[-RDP.]_ 

_Proof._ This corollary follows directly from Thm. 19 with the worst-case sensitivities _g[∗]_ and _h[∗]_ . 

## **C Scalable distributed learning** 

Our distributed learning extension for S-BDT builds on prior work [24]. We train a global ensemble _E_ that is shared amongst _k_ users with distinct private training datasets _D_ 1 _, D_ 2 _, ..., Dk_ . We assume the existence of a secure bulletin board that provides each user with the same set of hyperparameters. We utilize secure aggregation [8, 6], a protocol for privately computing the sum of vectors _A_ =[�] _[k] u_ =1 _[W][u]_[.][The][full][protocol][of][distributed][S-BDT][is] described in Alg. 7. 

Every user _u_ ( _u_ = 1 _,_ 2 _, ..., k_ ) receives the hyperparameters from the bulletin board (line 1) and sets up the accounting (line 3). 

For the initial classifier (from line 4), the user computes DP releases of the sum of labels and the dataset size. This is a slight variation of Alg. 3 ( `DPInitialScore` ) where 

**Algorithm 7:** Distributed-TrainSBDT 

||**Input:** _Du_ : private training dataset of user _u_|||||
|---|---|---|---|---|---|
||**:**_k_ : number of users|||||
||**:**(_r_1_, r_2) : noise weights for leaf value|||||
||**:**_g∗, h∗, m∗_: clipping bounds on gradients,|||||
||Hessians and labels|||||
||**:**_λ, β_ : regularization parameters|||||
||**:**(_T_regular_, T_extra) : number of rounds and|||extra||
||rounds|||||
||**:**_d_ : depth of trees|||||
||**:**_γ_ : subsampling ratio|||||
||**:**_m_ : number of features of _Du_|||||
||**:**(_v_(1)<br>min_, v_(1)<br>`max`_, ..., v_(_m_)<br>min_, v_(_m_)<br>`max` ) : feature value|||ranges||
|**1**|((_r_1_, r_2)_, g∗, λ, T_regular_, T_extra_, d, γ_) =|||||
||`BulletinBoard(`_’hyperparameters’_`)`|||||
|**2**|_T_max =_T_regular+_T_extra|||||
|**3**|ˆ_α, σ_2<br>leaf_, ρ_(ˆ_α_) = `Initialize(`_α_`max`_,_|||||
||(_εtrees, δtrees_)_, εinit, γ_`)`|||||
|**4**|(ds_u,_sum_u_) =`DPInitScore(`_D, m∗, εinit_`)`|||||
|**5**|ds= `SecureAggregation(`_ds_1_, ..., dsu, ..., dsk_`)`|||||
|**6**|sum= `SecureAggregation(`_sum_1_, ..., sumu, ..., sumk_`)`|||||
|**7**|init0 =sum_/_ds|||||
|**8**|_E_ = (init0)|||||
|**9 **|**for** _t_= 1 _to Tmax_ **do**|||||
|**10**|**for** _i_= 1 _to |Du|_ **do**|||||
|**11**|_ρ_(_i_)<br>_t_ (_α_) =_aγ_(_α,_<br>_α_<br>_σ_2<br>leaf _·_<br>�<br>_r_1_·|hi|_2<br>(_h∗_)2<br>+ _r_2_·|gi|_2<br>(_g∗_)2|||�|) ;|
||`// by Thm. 19 (for RDP: Cor. 20)`|||||
|**12**|_Du,t_ = (_xi_ :_F_ˆ_α,ρ_(ˆ_α_)(_ρ_(_u,i_)<br>1<br>_, ρ_(_u,i_)<br>2<br>_, . . . , ρ_(_u,i_)<br>_t_|) =||||
||CONT) `// by`<br>`Thm. 11`|||||
|**13**|tree_t_ = `CompleteTree()`|||||
|**14**|**for** _each split_ (_i, v_) _in treet_ **do**|||||
|**15**|_i_=_⌈_`PublicUniformSampling(`[0_, m_)`)`_⌉_|||||
|**16**|_v_ =`PublicUniformSampling(`[0_, v_(_i_)<br>`max` _−v_(_i_)<br>min)`)`_−_|||||
||_v_(_f_)<br>min|||||
|**17**|tree_t_ = `TrainSingleTree(`_treet, Dt, d,_|||||
||_σ_2<br>_leaf, g∗, h∗,_(_r_1_, r_2)_, λ, β, E_`)`|||||
|**18**|_Vu_ = list()|||||
|**19**|_Wu_ = list()|||||
|**20**|**for** _l_ = 1 _to_ 2_d_ **do**|||||
|**21**|let _vl, wl_ be gradient sum / Hessian sum||of||leaf _l_|
||in tree_t_|||||
|**22**|`Append(`_Vu, vl_`)`|||||
|**23**|`Append(`_Wu, wl_`)`|||||
|**24**|_A_= `SecureAggregation(`_W_1_, . . . ,_**Wu**_, . . . ,_|_Wk_`)`||||
|**25**|_B_ = `SecureAggregation(`_V_1_, . . . ,_**Vu**_, . . . , Vk_`)`||||**for**|
||_l_ = 1 _to_ 2_d_ **do**|||||
|**26**|`SetLeaf(`_treet, l, A_[_l_]_/B_[_l_]`)`|||||
|**27**|_E_ = (init0_,_tree1_,_tree2_, . . . ,_tree_t_)|||||
|**28 **|**return** E|||||



usually the sum would already be divided by the dataset size. All users synchronize and invoke `SecureAggregation` [8, 6] with fixed precision, for aggregating the dataset size first and then the label sum. The initial score is then built by dividing the aggregated label sum by the overall dataset size and added to the ensemble. 

The user commences _T_ `max` rounds of training, and starts a single training round by updating the individual Rényi DP privacy losses for all its data points (line 11) and then filtering out those data points that have expended all their privacy budget (line 12). Next, the user locally initializes tree _t_ for the current round _t_ (line 13) with arbitrary or even undefined splits, and then synchronizes with the other 

20 

users and utilizes public uniform sampling [30, Protocol 1] to collaboratively and verifiably sample uniformly random features and feature values for the splits of tree _t_ : For the random feature (line 15) chosen from _m_ features we draw a randomly uniform sample _i[′]_ from [0 _, m_ ) and then select the feature _⌈i[′] ⌉_ . For the feature value (line 16) of a numerical feature _i_ with feature range [ _v_ min[(] _[i]_[)] _[, v]_ `max`[(] _[i]_[)][)][we][draw] a randomly uniform sample _v[′]_ from [0 _, v_ `max`[(] _[i]_[)] _[−][v]_ min[(] _[i]_[)][)][and] then select the feature value _v[′] − v_ min[(] _[i]_[)][.][If][the][feature] _[i]_[is] categorical we assume _v_ `max`[(] _[i]_[)][=] _[ c]_[be][the][number][of][different] values of feature _i_ and _v_ min[(] _[i]_[)][= 0][.][We then draw a randomly] uniform sample _v[′]_ from [0 _, c_ ) and select the categorical feature value with index _⌈v[′] ⌉_ . 

The user then locally only adjusts the leaves of tree _t_ with its local share of sensitive data _Du_ (line 17). Note that this is a slight variation of Alg. 4 ( `TrainSingleTree` ): When already given a random tree, Alg. 4 must leave the splits untouched and only train the leaves of the given tree. 

All users must now synchronize again to utilize `SecureAggregation` [8] [6] with fixed precision for collaboratively generating leaf values for the tree. User _u_ generates vectors _Vu, Wu_ to contain the gradient sum and Hessian sum of all its local leaf values (line 22 and 23). The user then calls `SecureAggregation` twice (line 24 and 25) to privately compute 

**==> picture [120 x 13] intentionally omitted <==**

containing the collaboratively generated gradient sum and Hessian sum. User _u_ then sets the leaf values of its local tree (line 26) and finally adds this tree to the ensemble. 

21 

