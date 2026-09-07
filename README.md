# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# Fair Marketing Analytics

Analysing fairness, vulnerability and governance in automated financial product targeting.

**Live dashboard:**
[tableau dashboard](https://public.tableau.com/views/FairMarketing/GovernanceRecommendation?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) 
---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Project Management](#project-management)
- [Project Hypotheses and Validation](#project-hypotheses-and-validation)
- [Hypothesis Validation Summary](#hypothesis-validation-summary)
- [Project Plan](#project-plan)
- [Data Collection, Cleaning and Preparation](#data-collection-cleaning-and-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Modelling](#machine-learning-modelling)
- [Rationale to Map the Business Requirements to the Data Visualisations](#rationale-to-map-the-business-requirements-to-the-data-visualisations)
- [Analysis Techniques Used](#analysis-techniques-used)
- [Ethical Considerations](#ethical-considerations)
- [Dashboard Design](#dashboard-design)
- [Key Findings](#key-findings)
- [Unfixed Bugs](#unfixed-bugs)
- [Development Roadmap](#development-roadmap)
- [Deployment](#deployment)
- [Main Data Analysis Libraries](#main-data-analysis-libraries)
- [Use of Generative AI](#use-of-generative-ai)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)
- [Reflections](#Reflections) 

---

## Overview

This project examines a direct marketing campaign in which a bank telephoned existing
clients to sell a fixed-term deposit product. It asks three questions: which clients
subscribed, whether a model could be built to target future campaigns, and whether that
model should be deployed.

The third question is the subject of the project. A targeting model can be accurate,
commercially effective, and still produce outcomes a conduct function would not approve.
This analysis measures that gap rather than assuming it away.

### Framing

The dataset records telephone campaigns run by a Portuguese banking institution between May
2008 and November 2010. This project treats it as a proxy for a UK retail bank's outbound
campaign and assesses the resulting model against UK law and FCA rules. The framing is
stated because it governs the legal analysis: references to the Consumer Duty, PECR and UK
GDPR describe how such a system would be assessed if deployed by a UK-regulated firm, not
findings about any actual firm.

### Audiences

| Audience | Role | Requirement |
|---|---|---|
| Technical | Marketing analytics lead | Test statistics, effect sizes, model metrics, leakage evidence, subgroup performance |
| Non-technical | Consumer Duty / conduct risk lead | Which clients the model targets, whether that creates foreseeable harm, what controls would be required before approval |

---

## Project Structure

```
fair-marketing-analytics/
├── app.py                          Streamlit entry point
├── app_pages/                      One module per dashboard page
├── src/                            Reusable loaders and fairness functions
├── jupyter_notebooks/
│   ├── 01-ETL.ipynb                Extract, clean, transform
│   ├── 02-EDA.ipynb                Profiling and hypothesis testing
│   ├── 03-Data_Visualisation.ipynb Charts for the dashboard
│   └── 04-Machine_Learning.ipynb   Modelling and fairness evaluation
├── Data_Set/
│   ├── raw_data/                   Source dataset as downloaded
│   ├── clean_data/v1/              Cleaned dataset
│   ├── outputs/v1/                 Results tables and figures
│   └── models/v1/                  Fitted model pipelines
├── images/                         Screenshots for documentation
├── requirements.txt
└── README.md
```

Outputs are written to versioned directories created in code at the start of each notebook,
so a rerun cannot silently overwrite the artefacts a previous version produced.

---

## Dataset Content

**Source:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5K306

**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), which permits
sharing and adaptation for any purpose provided appropriate credit is given.

**File used:** `bank-additional-full.csv` — 41,188 records, 20 input variables, ordered by
date from May 2008 to November 2010. Semicolon-delimited.

The richer 20-variable file was chosen over the older 17-variable version for its
macroeconomic context fields, accepting the loss of the `balance` field.

### Fields

**Client:** age, job, marital status, education, credit default status, housing loan,
personal loan

**Campaign:** contact channel, month, day of week, call duration, number of contacts this
campaign, days since previous contact, number of previous contacts, previous campaign
outcome

**Macroeconomic context:** employment variation rate, consumer price index, consumer
confidence index, Euribor 3-month rate, number of employees

**Target:** whether the client subscribed to the term deposit

### Data protection position

The dataset contains no direct identifiers. It does contain age, occupation, marital status,
education and financial product holdings — information that, in a live setting, would
constitute personal data under UK GDPR and would require a lawful basis for processing.
Section 4 of Notebook 01 records the minimisation decisions taken on that basis.

---

## Business Requirements

| ID | Requirement |
|---|---|
| BR1 | Understand the composition of the campaign and the distribution of outcomes |
| BR2 | Identify which client and campaign characteristics are associated with subscription |
| BR3 | Build a targeting model free of leakage and measure whether its targeting falls disproportionately on potentially vulnerable groups |
| BR4 | Communicate findings, limitations and governance implications to both a technical and a non-technical audience |

BR3 is the central requirement. The analysis concludes with a documented recommendation on
whether the model should be deployed.

---

## Project Management

Work was tracked on a GitHub Project board with Backlog, In Progress and Done columns, with
one card per project phase.

---

## Project Hypotheses and Validation

Five hypotheses were defined before testing. Each was assigned a statistical method in
advance.

Categorical associations were tested using the chi-square test of independence with Cramér's
V as the effect size. Comparisons of a numeric variable across the two outcome groups used
the Mann-Whitney U test, chosen over the t-test because the campaign variables are heavily
skewed and not normally distributed, with rank-biserial correlation as the effect size.
Significance was assessed at alpha = 0.05.

### Why effect sizes are reported throughout

With 41,176 records, the chi-square test returns a significant result for differences far too
small to act on. Reporting p-values alone would overstate every finding. Cramér's V is
interpreted here as negligible below 0.10, small to 0.20, moderate to 0.30 and large above.

### H1 — Subscription rate differs by age band

**Supported.** Chi-square, p = 1.35e-258, Cramér's V = 0.170 (small).

Clients aged 60 and over subscribed at 39.60% and those under 25 at 23.99%, against 10.05%
for the 25 to 59 group. The effect size is small in absolute terms, but the disparity is the
largest in the dataset and is the pattern a model trained on this data learns most strongly.
Age is a protected characteristic under the Equality Act 2010.

### H2 — Subscription rate does not differ by education level

**Not supported.** Chi-square, p = 2.65e-35, Cramér's V = 0.067 (negligible).

Framed as a null hypothesis. Rates rise consistently from 7.82% for basic 9y schooling to
13.72% for university degree, so the hypothesis of no difference is rejected. The negligible
effect size indicates education explains very little of the variation on its own. Education
is not a protected characteristic but functions as a proxy for socioeconomic status.

### H3 — Subscription rate differs by housing loan status

**Supported statistically, not practically.** Chi-square, p = 0.0196, Cramér's V = 0.012
(negligible).

Rates are 11.62% and 10.88%, a difference of 0.74 percentage points. This hypothesis is
retained because it demonstrates the central methodological point of the analysis: at this
sample size, statistical significance can be achieved by differences too small to act on.
Reporting the p-value alone would have supported the claim that housing loan status
"significantly affects" subscription, which would be technically defensible and substantively
misleading.

### H4 — Repeated contact is associated with lower subscription

**Supported.** Mann-Whitney U, p = 3.63e-38, rank-biserial r = 0.110.

A conduct hypothesis as much as a commercial one. Repeatedly telephoning clients who continue
to decline raises a fair treatment question independently of whether it is effective.

### H5 — Call duration cannot serve as a legitimate predictor

**Supported. Leakage confirmed.** Mann-Whitney U, p < 0.001, rank-biserial r = −0.637,
point-biserial correlation 0.405.

Median duration was 449 seconds for subscribers against 164 seconds for non-subscribers, and
all four zero-second calls resulted in no subscription.

The dataset documentation states that duration should be discarded for a realistic predictive
model. This hypothesis tested that statement rather than accepting it. Duration is not
available when a targeting decision is made and is only known once the outcome is effectively
determined — it encodes the answer rather than predicting it. It is excluded from the
deployment model, and a comparison model retaining it quantifies the distortion it introduces.

---

## Hypothesis Validation Summary

| ID | Hypothesis | Test | p-value | Effect size | Interpretation | Outcome |
|---|---|---|---|---|---|---|
| H1 | Subscription rate differs by age band | Chi-square | 1.35e-258 | V = 0.170 | Small | **Supported** |
| H2 | Rate does not differ by education level | Chi-square | 2.65e-35 | V = 0.067 | Negligible | **Not supported** |
| H3 | Rate differs by housing loan status | Chi-square | 0.0196 | V = 0.012 | Negligible | **Supported** (not practically) |
| H4 | Contact count differs between groups | Mann-Whitney U | 3.63e-38 | r = 0.110 | Small | **Supported** |
| H5 | Duration cannot serve as a predictor | Mann-Whitney U | < 0.001 | r = −0.637 | Leakage confirmed | **Supported** |

Three of the five tests returned p < 0.05 with an effect size below the negligible threshold
or close to it. Only H1 and H5 describe differences large enough to act on.

---

## Project Plan

| Phase | Stage | Activity |
|---|---|---|
| 1 | Implementation | Business scoping, audience definition, ethics framing |
| 2 | Implementation | Repository setup, environment, versioned output structure |
| 3 | Implementation | ETL: quality assessment, missing value treatment, feature engineering |
| 4 | Implementation | Exploratory analysis and hypothesis testing |
| 5 | Implementation | Model development and fairness evaluation |
| 6 | Implementation | Dashboard design and build |
| 7 | Implementation | Deployment |
| 8 | Evaluation | User testing, accessibility review, iteration |
| 9 | Evaluation | Documentation and submission |

[kanban board](https://github.com/users/Haroon-Code-2026/projects/3)
### Maintenance and updates

Were this deployed, the following would be required rather than optional.

**Monitoring.** False positive rate by age band would be monitored as a live control with a
defined threshold and escalation route, not as a one-off pre-deployment test. The disparity
identified in this analysis arises through proxy variables, so it can re-emerge after
retraining with no change to the feature set.

**Retraining.** Over half the model's predictive power derives from macroeconomic variables
describing conditions between 2008 and 2010. Any deployment would require retraining on
current data and revalidation of the fairness metrics at each retrain.

**Feature governance.** Exclusion of `duration` would be recorded in the model specification
rather than left to the modeller, since its removal is not obvious from the data alone.

---

## Data Collection, Cleaning and Preparation

Notebook: `01-ETL.ipynb`

### Quality assessment

41,188 records, 21 columns, 12 duplicate rows. No null values in the conventional sense —
missing information is recorded as the string `unknown`, which pandas reads as a valid
category.

| Field | `unknown` | % |
|---|---|---|
| default | 8,597 | 20.87 |
| education | 1,731 | 4.20 |
| housing | 990 | 2.40 |
| loan | 990 | 2.40 |
| job | 330 | 0.80 |
| marital | 80 | 0.19 |

`housing` and `loan` are missing on exactly the same 990 records, indicating a structural
collection failure rather than random missingness.

### Decisions taken and why

**`unknown` converted to null, not dropped.** Removing those records would silently exclude
clients whose data was never captured. The pattern of missingness is itself relevant to the
fairness assessment. Records with missing values do not subscribe at unusual rates, so the
missingness does not appear to bias the outcome.

**`default` replaced with a disclosure flag.** The field records three "yes" values across
41,188 records against 32,588 "no" and 8,597 "unknown". It carries almost no information
about actual credit default. What it captures is whether the client answered the question.
Retained as a predictor, a model would learn from non-disclosure rather than from credit
risk, penalising clients who declined to answer. It was replaced with `default_disclosed`.

**`pdays` = 999 recoded.** In this dataset 999 indicates no previous contact. Left as a
number it would be read as an extremely long gap and would distort any model. It was replaced
with a `contacted_before` flag and the numeric field set to null where no prior contact
occurred.

**Demographics retained but not modelled.** Age, age band, job, education and marital status
are kept in the cleaned dataset because they are required to measure whether the model's
targeting falls evenly across groups. They are excluded from the model's feature set. This is
a data minimisation decision applied at the point of use rather than at the point of
collection.

**Age banded** into under 25, 25 to 59, and 60 and over — the groups most relevant to a UK
conduct assessment, where younger and older customers are more likely to be treated as
potentially vulnerable.

**Column names standardised** to lowercase with underscores, since five macroeconomic fields
used dots in their names.

**Output:** `Data_Set/clean_data/v1/bank_marketing_cleaned.csv` — 41,176 records.

---

## Exploratory Data Analysis

Notebook: `02-EDA.ipynb`

The overall subscription rate is 11.3%, a class ratio of roughly 8 to 1.

### Subscription rate by age band

| Age band | Contacted | Subscribed | Rate |
|---|---|---|---|
| 60 and over | 1,192 | 472 | 39.60% |
| Under 25 | 1,067 | 256 | 23.99% |
| 25 to 59 | 38,917 | 3,911 | 10.05% |

The two age groups most likely to be treated as potentially vulnerable under UK conduct rules
are the two the campaign converted most successfully.

### Subscription rate by education

| Education | Contacted | Rate |
|---|---|---|
| University degree | 12,164 | 13.72% |
| Professional course | 5,240 | 11.35% |
| High school | 9,512 | 10.84% |
| Basic 4y | 4,176 | 10.25% |
| Basic 6y | 2,291 | 8.21% |
| Basic 9y | 6,045 | 7.82% |

The `illiterate` category contains 18 records and is not interpreted.

### Subscription rate by job

Students at 31.43% and retired clients at 25.26% lead; blue-collar workers at 6.90% are
lowest. These groups overlap substantially with the outer age bands and are not treated as
independent findings.

### Contact volume against success rate

May carried the highest contact volume of any month at 13,767 clients and the lowest
subscription rate at 6.4%. The four highest-rate months — March at 50.5%, December at 48.9%,
September at 44.9% and October at 43.9% — together account for 2,015 contacts, under 5% of
the campaign.

This is not evidence that calling in March causes subscription. Low-volume, high-rate months
are consistent with targeted follow-up against a pre-qualified list; high-volume months are
consistent with untargeted outreach. Volume and selectivity are confounded. What the finding
does establish for BR1 is that the campaign spent the majority of its contact effort in its
least productive periods.

### Age and occupation interact

Among clients aged 60 and over, subscription rates are high across nearly every occupation
present in sufficient numbers: 46.3% housemaid, 42.6% retired, 41.5% administrative, 37.9%
management, 30.6% technician. The same occupations sit between 7.6% and 12.6% in the 25 to 59
band. Blue-collar workers are lowest in every band, but their 60-and-over rate of 18.6% is
still nearly three times their 25 to 59 rate of 6.8%.

The consequence is that age carries signal independently of occupation and occupation carries
signal independently of age. Removing one from a model would not remove the pattern the other
encodes. This prediction was tested directly in Notebook 04.

---

## Machine Learning Modelling

Notebook: `04-Machine_Learning.ipynb`

### Approach

Two feature sets, one algorithm. Random Forest was used throughout because the exploratory
analysis showed age and occupation interacting, and a model capable of representing
interactions is appropriate. Using one algorithm across both sets means the only difference
between them is the leaked field.

**Set A** retains `duration`. Built solely to quantify what leakage does to apparent
performance. Never a deployment candidate.
**Set B** excludes `duration` and all demographic fields. The deployment candidate.

A 75/25 stratified split preserves the 11.3% positive rate in both sets. The full feature
frame is split once and subset per model, so both models are evaluated on identical clients
and the demographic columns remain available in the test set for fairness measurement.

`class_weight='balanced'` is applied. Without it the 8:1 imbalance drives the model towards
the majority class, producing high accuracy and near-zero recall.

### Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (predict "no" for all) | 0.8873 | 0.000 | 0.000 | 0.000 | 0.500 |
| Set A — retains duration | 0.8860 | 0.4966 | 0.8888 | 0.6372 | 0.9484 |
| **Set B — deployment candidate** | **0.8380** | **0.3700** | **0.6233** | **0.4644** | **0.7939** |

Set B's accuracy is *lower* than the baseline's. This is the expected result and the reason
accuracy was excluded as a headline metric at the outset: the baseline achieves 88.7% by
identifying nobody.

On the 10,294 held-out clients, Set B recommends contacting 1,954 and correctly identifies
723 of the 1,160 who subscribed. Contacting 19% of the list captures 62% of subscribers, and
37.0% precision is a 3.3-fold improvement on the 11.3% base rate.

| | Predicted: no contact | Predicted: contact |
|---|---|---|
| **Did not subscribe** | 7,903 | 1,231 |
| **Subscribed** | 437 | 723 |

### The cost of leakage

Removing one leaked field costs 0.155 of ROC-AUC and over a quarter of recall. None of Set A's
advantage is real. A model risk process reviewing headline metrics alone would have approved a
model reporting 0.95 ROC-AUC that cannot function in production, because call duration is
unknown at the moment a targeting decision is taken.

### What the model actually predicts

| Feature | Importance |
|---|---|
| euribor3m | 0.212 |
| campaign | 0.142 |
| nr_employed | 0.140 |
| emp_var_rate | 0.088 |
| cons_conf_idx | 0.053 |
| cons_price_idx | 0.040 |
| default_disclosed | 0.032 |
| poutcome_success | 0.031 |
| contacted_before | 0.029 |

Five macroeconomic variables account for approximately 53% of total feature importance. Over
half the model's predictive power derives from the economic environment rather than from any
characteristic of the client. The campaign ran through a period in which euro interbank rates
fell sharply and employment contracted; the model has substantially learned the relationship
between a specific economic episode and demand for fixed-term deposits. This limits
transferability severely.

---

## Rationale to Map the Business Requirements to the Data Visualisations

| Chart | Type | BR | Rationale |
|---|---|---|---|
| Outcome distribution | Bar | BR1 | The outcome is binary and the point is proportion. Establishes the 11.3% baseline against which every subgroup is judged. |
| Contact volume and rate by month | Bar + line, twin axes | BR1 | Two quantities share a category axis but differ by orders of magnitude, and the inverse relationship between them is the finding. |
| Subscription rate by age band | Bar with reference line | BR2, H1 | Shows absolute rates and departure from baseline together. Group sizes labelled because the outer bands are small. |
| Subscription rate by education | Horizontal bar | BR2, H2 | Category labels are long and the ordering is ordinal. Effect size placed in the title because the visual gradient over-reads. |
| Call duration by outcome | Box plot | BR3, H5 | Duration is heavily skewed; the comparison of interest is between distributions, not means. |
| Rate by job and age band | Heatmap | BR2 | Two categorical variables interacting across eleven categories would be unreadable as grouped bars. |
| Effect sizes across hypotheses | Horizontal bar | BR4 | Visualises method rather than data. Addresses the most likely misreading of the project. |
| Leakage comparison | Bar | BR3 | A single number comparison; the gap is the message. |
| Confusion matrix | Heatmap | BR3 | Standard representation; the false positive count is titled explicitly. |
| Feature importance | Horizontal bar | BR3 | Ranked magnitudes with long labels. |
| Fairness by age band | Grouped bar | BR3, BR4 | Three related rates per group must be compared within and across groups simultaneously. |

Charts that do not answer a stated business requirement were not produced.

---

## Analysis Techniques Used

**Chi-square test of independence** — for association between two categorical variables (H1,
H2, H3). Reported with Cramér's V.

**Mann-Whitney U** — for comparing a numeric variable across the two outcome groups (H4, H5).
Chosen over the t-test because the campaign variables are heavily skewed and not normally
distributed. Reported with rank-biserial correlation.

**Point-biserial correlation** — as a second measure of the duration relationship in H5.

**Random Forest classification** — chosen for its ability to represent the age-occupation
interaction identified in the exploratory analysis, and for native feature importances.

**Fairness metrics** — selection rate, true positive rate, false positive rate and precision
computed per subgroup, summarised as demographic parity difference, equal opportunity
difference and false positive rate difference.

### Limitations

- All findings are associational. Clients were not randomly assigned to groups, and
  differences may reflect who the bank chose to contact as much as how clients responded.
- Age band and job category overlap substantially and are not independent findings.
- The dataset records month of contact but not year, so month-based charts show seasonal
  pattern, not trend.
- Group sizes for the outer age bands in the test set are small (283 and 272), so rates
  reported for them carry meaningful uncertainty.
- The campaign spans the 2008 financial crisis, limiting how far patterns transfer to
  present-day conditions.

---

## Ethical Considerations

### Privacy and data minimisation

The dataset contains no direct identifiers. It does contain age, occupation, marital status,
education and financial product holdings, which in a live setting would constitute personal
data under UK GDPR and require a lawful basis.

Minimisation was applied at the point of use rather than only at collection. Demographic
fields were retained in the cleaned dataset but excluded from the model's feature set, on the
basis that they are necessary to *audit* the system for fairness but not necessary to *operate*
it. Removing them entirely would have made the fairness analysis in this project impossible —
a live risk in real deployments, where deleting protected characteristics is sometimes
mistaken for a privacy control when it primarily removes the ability to detect discrimination.

### Missing data as an equality issue

Credit default status is unrecorded for 20.87% of clients. The field most closely related to a
client's financial position is the least reliably captured. Retaining it as a predictor would
have meant the model learning from whether a client answered a question rather than from their
circumstances. It was converted to an explicit disclosure flag, and the residual concern —
that the model still assigns 0.032 importance to disclosure behaviour — is carried into the
deployment recommendation.

### Bias and fairness

Two fairness definitions were considered, because they answer different questions and cannot
generally both be satisfied.

**Demographic parity** asks whether groups are targeted at equal rates, ignoring whether
members of each group wanted the product. Enforcing it here would mean contacting clients aged
60 and over at the same rate as everyone else despite an observed 39.6% against 10.1%
difference in subscription.

**Equal opportunity** asks whether the model finds genuine subscribers equally well within each
group, permitting unequal targeting where underlying rates genuinely differ.

**Equal opportunity was adopted**, stated before the metrics were computed, on the grounds that
the model's purpose is to identify interested clients and the underlying rates are not equal.
That choice carries a consequence: it accepts that clients aged 60 and over will be contacted
disproportionately often. Whether that is acceptable is a conduct judgement, not a statistical
one.

**The model does not meet the standard adopted.** Equal opportunity difference on age band is
0.326 — it identifies 90.2% of genuine subscribers aged 60 and over but only 57.6% of those
aged 25 to 59.

### Fairness by age band

| Age band | n | Actual rate | Model targets | False positive rate | Finds subscribers |
|---|---|---|---|---|---|
| 60 and over | 283 | 43.5% | 80.9% | 73.8% | 90.2% |
| Under 25 | 272 | 22.8% | 40.8% | 29.1% | 80.7% |
| 25 to 59 | 9,739 | 10.0% | 16.6% | 12.0% | 57.6% |

Demographic parity difference 0.6435 · Equal opportunity difference 0.3260 · False positive
rate difference 0.6175

### What a false positive costs

A false positive is a client the model recommends contacting who does not subscribe. In a
commercial frame this is a wasted call. In a conduct frame it is unsolicited contact with
someone who did not want the product.

Among clients aged 60 and over who would not have subscribed, 73.8% are recommended for
contact. The equivalent figure for the 25 to 59 group is 12.0%. The model directs unwanted
contact at older clients at roughly six times the rate it does at the working-age majority.
Under FCA vulnerable customer guidance and the Consumer Duty obligation to avoid foreseeable
harm, that is not a neutral cost.

### The choice of fairness definition determines what is visible

| Measure | Age band | Education |
|---|---|---|
| Demographic parity difference | 0.6435 | 0.1324 |
| Equal opportunity difference | 0.3260 | 0.3367 |
| False positive rate difference | 0.6175 | 0.0923 |

On demographic parity, education appears close to even. On equal opportunity it is the worse
of the two attributes: the model finds 70.5% of genuine subscribers among clients with a
university degree and 36.8% among those with basic 9y schooling.

An assessment reporting demographic parity alone would have concluded that education presents
no fairness concern. That conclusion would have been wrong, and wrong in the direction that
favours deployment.

### Fairness through unawareness does not work

The deployment model contains no demographic inputs. It has never seen a client's age. It
nevertheless produces a demographic parity difference of 0.6435 across age bands.

The disparity is carried by proxies. Campaign fields including prior contact history, contact
channel and previous campaign outcome encode who the bank has approached before and how, and
that history is not independent of age. The exploratory analysis anticipated this: age and
occupation each carried signal independently of the other, indicating the pattern is
distributed across the data rather than confined to any one field.

Deleting protected characteristics from a feature set removes the ability to measure
disparity, not the disparity itself.

### Legal and regulatory considerations

Assessed as if the system were deployed by a UK-regulated firm.

**UK GDPR and Data Protection Act 2018.** A lawful basis would be required for processing.
Article 21 gives an absolute right to object to direct marketing — not a balancing test, an
absolute right — which any targeting system must be able to honour at the individual level.
Article 22 restricts solely automated decision-making producing legal or similarly significant
effects; a marketing contact decision is unlikely to meet that threshold, but the boundary
would need to be assessed rather than assumed, particularly where the model's output
determines access to a product.

**Privacy and Electronic Communications Regulations 2003.** Live marketing calls require
screening against the Telephone Preference Service and respect for prior objections. A model
that ranks clients by propensity does not itself address consent, and the two must be applied
in sequence rather than in parallel.

**FCA Consumer Duty.** The cross-cutting obligation to avoid causing foreseeable harm is the
provision most directly engaged by the false positive findings. The consumer understanding and
consumer support outcomes would also apply to how a fixed-term product is explained to a
client contacted on the model's recommendation.

**FCA vulnerable customer guidance.** Age alone does not make a customer vulnerable, and this
analysis does not claim it does. But a targeting system that concentrates unsolicited contact
on the over-60s at six times the rate of other groups would require documented justification
and monitoring rather than passive acceptance.

**Equality Act 2010.** Age is a protected characteristic. Indirect discrimination arises where
a neutral practice puts a group at particular disadvantage without objective justification. A
model containing no age variable can still produce age-differentiated outcomes through proxies,
as demonstrated above.

**ICO guidance on AI and data protection.** Fairness, transparency and explainability, and the
expectation that fairness is monitored across the lifecycle rather than certified once.

### Social implications

The findings describe a commercially rational campaign. Older clients subscribed at four times
the rate of working-age clients, so a model optimising for subscription targets them. Nothing
in the data is anomalous.

That is the point. The harm, if it arises, is not a defect in the model — it is the model
working correctly against an objective that does not include conduct constraints. Systems of
this kind concentrate contact where response is highest, and where response correlates with
groups more likely to be vulnerable, that concentration is a foreseeable outcome rather than
an accident. Controls have to be imposed from outside the optimisation, because the
optimisation will not generate them.

---

## Dashboard Design

Built in Tableau Public across six dashboards. Ten worksheets carry the exploratory
analysis; the model results and ethics reasoning are presented as structured text, since
that work was done in Python rather than in Tableau.

### Serving two audiences

Criterion 2.1 requires both technical metrics and simplified summaries in the same view.
Every chart carries three layers arranged by position and weight:

- a plain-English takeaway above it, in bold, stating the finding in one sentence
- the chart itself, with all marks labelled
- the supporting statistics below it, in smaller grey text — test statistic, p-value,
  effect size, group sizes

A reader who reads only the bold lines receives the complete argument. A reader who wants
the evidence finds it directly beneath each chart without leaving the view.

Sheet titles state findings rather than naming variables. "Clients aged 60 and over
subscribed at nearly four times the rate of the working-age majority" rather than
"Subscription rate by age band".

### Accessibility

- Okabe-Ito colourblind-safe palette throughout, matching the notebook figures
- Every mark carries a value label, so no chart depends on colour to convey meaning
- Fixed dashboard size of 1200 x 900, checked at 100% zoom
- Manual sort applied to age band, education and month, since alphabetical ordering
  produces a misleading sequence for all three
- Group sizes displayed wherever a rate is shown
- Navigation objects on every dashboard so a reader can move through the sequence without
  using the tab strip

### Decisions worth noting

The `illiterate` education category is excluded from the education chart. It contains 18
records, and a rate calculated from a cell that size would suggest a precision the data
does not support. The exclusion is stated on the dashboard rather than applied silently.

Heatmap cells containing fewer than 30 clients are left blank. A heatmap invites the eye to
compare every cell equally, and an unsuppressed cell of five clients would read as a
finding.

Call duration is shown as a median rather than a mean, matching the figures reported in the
hypothesis testing. Duration is heavily skewed, so the median is the more honest summary.


---

## Key Findings

1. **The campaign converted older and younger clients far better than the working-age
   majority.** 39.6% for the over-60s and 24.0% for the under-25s, against 10.1% for the 25 to
   59 group.

2. **A single leaked field inflated apparent model performance from 0.79 to 0.95 ROC-AUC.**
   Call duration is not known when a targeting decision is made. The inflated figure would
   have passed a review of headline metrics.

3. **The deployment model contains no demographic inputs and still targets by age.**
   Demographic parity difference of 0.6435 arises entirely through proxy variables.

4. **The model fails the fairness standard adopted before testing.** Equal opportunity
   difference of 0.326 on age and 0.337 on education.

5. **It directs unwanted contact at older clients at six times the rate of the working-age
   majority.** 73.8% false positive rate against 12.0%.

6. **Which fairness definition you choose determines what you can see.** Education looks even
   on demographic parity (0.1324) and is the worse of the two attributes on equal opportunity
   (0.3367).

7. **Over half the model's predictive power is macroeconomic.** Five economy-level variables
   account for approximately 53% of feature importance, limiting transferability.

8. **Statistical significance is not importance.** Three of five hypotheses returned p < 0.05
   with negligible or near-negligible effect sizes.

**Recommendation: do not deploy in its current form.** Conditions for reconsideration are set
out in Section 5 of `04-Machine_Learning.ipynb` and summarised in the Development Roadmap
below.

---

## Unfixed Bugs

I had several issues/bugs along the way including:
- The subscibed field being read as measure as defaul
- Text overflowing in dashboard write ups from 4-6
- Two graphs looked odd so I re-ran after consulting Claude and tweaked columns/rows
---

## Development Roadmap

**Conditions under which the model could be reconsidered**

- Recalibrate decision thresholds by group and re-measure equal opportunity difference. The
  current disparity arises at a single global threshold.
- Cap contact frequency, particularly for clients aged 60 and over. H4 found repeated contact
  associated with lower subscription, so a cap is supported by the data as well as by conduct
  considerations.
- Remove `default_disclosed` from the feature set — it rewards disclosure rather than
  measuring risk.
- Retrain on a period spanning more than one interest rate environment, or remove the
  macroeconomic block and accept reduced performance in exchange for a model whose signal
  derives from client and campaign characteristics.
- Permanently exclude `duration` in the model specification rather than leaving it to the
  modeller.
- Monitor false positive rate by age band as a live control with a defined threshold and
  escalation route.

**Analytical extensions**

- Threshold optimisation with an explicit fairness constraint, to quantify the recall cost of
  meeting equal opportunity.
- Intersectional fairness analysis across age and education jointly.
- Comparison against a gradient boosting model to test whether the fairness profile is
  algorithm-specific or inherent to the data.

---

## Main Data Analysis Libraries

| Library | Use |
|---|---|
| pandas | Data loading, cleaning, aggregation |
| numpy | Numerical operations |
| scipy.stats | Chi-square, Mann-Whitney U, point-biserial correlation |
| scikit-learn | Pipelines, preprocessing, Random Forest, evaluation metrics |
| matplotlib | Chart construction |
| seaborn | Heatmaps and statistical plotting |
| joblib | Model serialisation |
| streamlit | Dashboard |

---

## Use of Generative AI

I used claude, not Claude Code, to help me with errors, write ups and validating results of my graphs.

---

## Credits

**Dataset**
Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5K306. Licensed under CC BY 4.0.

**Associated paper**
Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank
telemarketing. *Decision Support Systems*, 62, 22–31.

**Accessibility**
Chart colours use the Okabe-Ito colourblind-safe palette.

## Reflection

### Choosing the dataset took longer than it should have

The project began with a UK financial services dataset in mind, on the reasoning that a UK
regulatory analysis needs UK data. That proved harder than expected. Almost all
individual-level UK financial data with demographic variables — the Wealth and Assets
Survey, the Family Resources Survey, Financial Lives microdata — sits behind a UK Data
Service licence that prohibits redistribution, and the data has to be committed to a public
repository for the dashboard to read it. What remains openly available in the UK is
aggregate, firm-level or geographic, and none of it supports a fairness analysis about
people.

Several datasets were scoped and discarded before settling on the UCI Bank Marketing data
with an explicit framing device. That cost time, but the constraint turned out to be worth
understanding rather than working around: the reason open UK microdata is scarce is
precisely the privacy protection this project is about.

### The most useful decisions were made before the analysis started

Two choices made early did more work than anything that came later.

The first was framing two hypotheses as nulls — testing for the absence of disparity rather
than for its presence. That removed the temptation to go looking for a finding, and it meant
a "not supported" result was informative rather than a dead end.

The second was adopting equal opportunity as the fairness standard and writing it down
before computing any metrics. When the model then failed that standard by 0.326, the result
was evidence rather than an argument. Had the standard been chosen afterwards, the honest
options would have been to report a failure I had set myself up to find, or to pick the
measure that flattered the model. Choosing first closed off the second option.

### The result I did not expect

The model was built without age, education, job or marital status. It has never seen a
client's age. I expected the demographic disparity to shrink substantially as a result.

It produced a demographic parity difference of 0.6435 — effectively unchanged. The
disparity travels through prior contact history and campaign fields that encode who the bank
approached before.

In hindsight the visualisation work had already predicted this. The job-by-age heatmap
showed age and occupation each carrying signal independently of the other, which is exactly
the condition under which removing one field fails to remove the pattern. I did not connect
the two at the time, and only recognised the relationship when the fairness metrics came
back. The lesson is about sequencing: the exploratory finding was the more valuable of the
two, and it was sitting in front of me two notebooks earlier.

### Where I would spend more time

**Threshold tuning.** The disparity was measured at a single global decision threshold.
Group-specific thresholds might reduce it materially, at a stated cost to overall recall,
and quantifying that trade-off would make the recommendation more useful than a refusal.
This is documented as a condition for reconsideration rather than tested.

**Intersectional analysis.** Fairness was measured across age and education separately. The
group most likely to be affected — older clients with lower educational attainment — was
never examined as a group. Single-attribute fairness analysis can pass while the
intersection fails, and this project would not have detected that.

### What I would do differently

Write the data quality assessment before writing any hypotheses. The `default` field turned
out to record three positive cases in 41,188 records, and the fact that it functioned as a
disclosure indicator rather than a risk indicator was a finding in its own right. I
discovered it during cleaning rather than during scoping, which meant reworking part of the
feature set afterwards.

Commit more granularly during the analysis phase. Commits during the ETL and modelling work
covered more ground than they should have, which makes the development history harder to
read than it needs to be.





