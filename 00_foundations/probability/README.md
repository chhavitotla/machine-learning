# Probability

## The Intuition

Probability is the bookkeeping system for not knowing things. Machine learning is soaked in it because every dataset is a sample, every label is possibly noisy, and every prediction is a bet. Three results do most of the heavy lifting. The **Law of Large Numbers** says averages calm down: flip a biased coin enough times and the running average of heads locks onto the true bias — this is why "more data" works at all. The **Central Limit Theorem** is stranger and better: take averages of *anything* — coin flips, dice, ball-bearings bouncing off pegs — and the averages themselves pile up into the same bell-shaped curve, every time. Nature has a favorite distribution and it's the Gaussian, which is why it haunts every loss function and confidence interval. And **Bayes' rule** is the law of changing your mind: it tells you exactly how much a piece of evidence (a positive medical test, a spam-looking word) should shift your belief — and its most famous lesson is that humans get this badly wrong by ignoring base rates. A 99%-accurate test for a 1-in-1000 disease leaves a positive-testing patient with only a ~9% chance of being sick. Run the numbers; they don't care about your intuition.

## The Math

The Law of Large Numbers:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \;\xrightarrow{\;n \to \infty\;}\; \mathbb{E}[X]$$

- $X_i$ — independent draws of the same random quantity (e.g. coin flips, 1 = heads)
- $\bar{X}_n$ — the average of the first $n$ draws
- $\mathbb{E}[X]$ — the true expected value (for a coin with bias $p$, just $p$)

Averages of more and more samples converge to the truth — noise cancels itself out, slowly.

The Central Limit Theorem:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \;\xrightarrow{\;n \to \infty\;}\; \mathcal{N}(0, 1)$$

- $\mu$ — the true mean of a single draw
- $\sigma$ — the true standard deviation of a single draw
- $\mathcal{N}(0,1)$ — the standard normal: the bell curve

Whatever shape the individual draws have, their *average* is approximately Gaussian with spread $\sigma/\sqrt{n}$ — uncertainty shrinks like $1/\sqrt{n}$, which is why halving your error bars costs 4× the data.

Bayes' rule:

$$P(D \mid +) = \frac{P(+ \mid D)\,P(D)}{P(+ \mid D)\,P(D) + P(+ \mid \neg D)\,P(\neg D)}$$

- $P(D)$ — the prior: prevalence of the disease before any test
- $P(+ \mid D)$ — sensitivity: how often the test catches a real case
- $P(+ \mid \neg D)$ — false positive rate, equal to $1 - \text{specificity}$
- $P(D \mid +)$ — the posterior: what you actually want to know

The denominator counts *every* way to test positive — truly sick people caught, plus healthy people falsely flagged — and when the disease is rare, the second crowd is bigger, dragging the posterior far below the test's "accuracy."

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Reading any model's confidence** — a classifier's softmax output is a probability distribution, calibration is checking whether "80% confident" means right 80% of the time, and the CLT is why averaging an ensemble tightens predictions.
- **A/B tests and error bars** — the $\sigma/\sqrt{n}$ shrinkage tells you how many users you need before a 2% lift is signal rather than noise; every significance test you'll ever run leans on the CLT.
- **Rare-event systems: fraud, disease, anomaly detection** — whenever the positive class is scarce, Bayes' rule is the difference between a useful alert and an alarm that's wrong 91% of the time; precision *is* the posterior.

## What breaks it

- **Non-independent samples** — the LLN and CLT assume independent draws; average correlated data (today's users overlap with yesterday's, time-series points echo each other) and your error bars are confidently, badly wrong.
- **Heavy tails** — the CLT needs finite variance; sample means of power-law data (viral posts, financial crashes, file sizes) converge slowly or never, so the bell curve you assumed undersells extreme events catastrophically.
- **Trusting the test, forgetting the prior** — deploy a 99%-accurate fraud model on transactions where fraud is 0.1% and most of your flags are false positives; teams that skip the Bayes arithmetic ship alert systems that get ignored within a week.

## 5 Interview Questions

**1. Conceptual — "Why does the Gaussian show up everywhere in ML?"**
Direct answer: the Central Limit Theorem — any quantity that is the sum or average of many small independent effects is approximately normal, and that describes measurement noise, aggregated user behavior, and initialized weight sums alike; mathematically it's also the maximum-entropy distribution for a fixed mean and variance, i.e. the least-assuming choice. Reason: convolution of distributions smooths toward the bell shape regardless of the ingredients, so nature keeps reconverging on it. Likely follow-up: *"Where does the Gaussian assumption hurt you?"* — heavy-tailed data: finance, virality, latencies. Squared-error losses (Gaussian likelihood in disguise) get wrecked by those outliers; switch to robust losses or model the tail explicitly.

**2. Mathematical — "A disease affects 1 in 1000 people. A test has 99% sensitivity and 99% specificity. You test positive — what's the chance you're sick?"**
Direct answer: about 9% — $P(D|+) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} = \frac{0.00099}{0.00099 + 0.00999} \approx 0.090$. Reason: in 100,000 people there are ~99 true positives but ~999 false positives; positive tests are dominated by the huge healthy majority leaking through the 1% false-positive crack. Likely follow-up: *"How do you raise that 9%?"* — raise the prior (test only symptomatic people) or retest: a second independent positive multiplies the evidence and pushes the posterior past 90%.

**3. Practical — "Your A/B test shows a 2% lift after 500 users. Ship it?"**
Direct answer: no — check the standard error first; with binary outcomes the noise per user is ~0.5, so after 500 users the uncertainty on the mean is $\approx 0.5/\sqrt{500} \approx 2.2\%$ per arm, bigger than the lift itself. Reason: the CLT says your measured lift is the true lift plus Gaussian noise of width $\sigma/\sqrt{n}$, and right now the noise swamps the signal. Likely follow-up: *"How many users do you need?"* — invert the formula: to resolve a 2% lift at ~95% confidence you need the standard error of the *difference* below ~1%, which works out to several thousand users per arm; be ready to sketch $n \approx (2\sigma/\delta)^2$.

**4. Gotcha — "My anomaly detector is 99% accurate, so 99% of its alerts are real, right?"**
Direct answer: wrong — accuracy is $P(\text{correct})$, but alert quality is $P(\text{real} \mid \text{alert})$, a conditional flipped the other way around, and Bayes says flipping it requires the base rate; with 0.1% anomalies a "99% accurate" detector's alerts are ~91% false. Reason: confusing $P(+|D)$ with $P(D|+)$ is base-rate neglect, the single most common probability error in production ML. Likely follow-up: *"What metric should you quote instead?"* — precision and recall (precision literally *is* the posterior $P(D|+)$), or the full precision-recall curve, since ROC curves also flatter rare-class models.

**5. System design — "Design a spam filter using probability from first principles."**
Direct answer: Naive Bayes — model $P(\text{spam} \mid \text{words}) \propto P(\text{spam}) \prod_i P(w_i \mid \text{spam})$, with word likelihoods counted from labeled mail, a prior from the observed spam rate, Laplace smoothing so unseen words don't zero out the product, and log-probabilities so the product becomes a stable sum; threshold the posterior to trade false positives against false negatives. Reason: it's trained by counting, updates online trivially, and explains its decision in terms a human can audit ("the word 'lottery' is 40× likelier in spam"). Likely follow-up: *"Where does 'naive' bite you?"* — the independence assumption: "Hong" and "Kong" are not independent evidence, so the model double-counts correlated words and becomes overconfident; fixes range from better features (bigrams) to calibrating the output probabilities.
