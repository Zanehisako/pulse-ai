# 🔗 Interaction-Aware Online Learning Agent

## The Problem You Identified

Your original agent learns **main effects only**:
```
prediction = w1*weather + w2*distance + w3*age + ...
```

But real donor behavior has **interactions**:
- `weather × distance` - Bad weather matters MORE for distant donors
- `age × donation_frequency` - Older regular donors are MORE reliable
- `holiday × reliability_score` - Holidays affect unreliable donors MORE

Your agent was **missing these multiplicative effects**!

---

## The Solution: Residual-Based Interaction Discovery

### How It Works

#### 1. **Track Prediction Errors (Residuals)**
After each prediction, we measure: `residual = predicted - actual`

#### 2. **Correlate Errors with Feature Products**
For all pairs of active features (f1, f2), we track:
```python
correlation(residual, f1 × f2)
```

Using exponential moving average:
```python
corr[f1,f2] = 0.99 * corr[f1,f2] + 0.01 * residual * (f1 * f2)
```

#### 3. **Promote Strong Correlations**
Every 2000 steps, find the pair with highest correlation:
```python
if |correlation| > threshold:
    activate_interaction(f1, f2)
    add_weight: w_interaction * (f1 × f2)
```

#### 4. **Learn & Prune Like Main Effects**
- Interactions get their own weights
- Updated via gradient descent
- Pruned if they don't contribute after grace period

---

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `interaction_search_interval` | 2000 | How often to search for new interactions |
| `interaction_threshold` | 0.03 | Min correlation to activate a pair |
| `grace_period` | 4000 | Steps before pruning (features & interactions) |
| `l1_penalty` (interactions) | 1.5× | Slightly stronger regularization |

---

## Example Timeline

```
Step 0:     Base features added (age, recency, donation_count, ...)
            → 45 candidate pairs formed

Step 2000:  First interaction search
            → Found: weather × distance (corr=0.087)
            ✅ Activated

Step 5000:  New feature: transfer_cost
            → 10 new candidate pairs (transfer_cost × others)

Step 6000:  Interaction search
            → Found: age × donation_frequency (corr=0.045)
            ✅ Activated

Step 8000:  Pruning check
            → weather × distance weight = 0.08 (STRONG)
            → holiday × bmi weight = 0.003 (WEAK)
            🗑️ Pruned holiday × bmi

Step 10000: Final model
            ✅ 12 main features
            ✅ 4 strong interactions
```

---

## Why This Approach Works

### ✅ Advantages
1. **Discovers interactions organically** - No need to specify them upfront
2. **Handles late-arriving features** - Interactions form as features join
3. **Prunes useless pairs** - Only keeps what improves predictions
4. **Computational efficiency** - Only tests O(n²) pairs, not O(n³) triplets

### ⚠️ Limitations
1. **Pairwise only** - Doesn't find 3-way interactions (weather × distance × age)
2. **Correlation ≠ Causation** - High correlation might be spurious
3. **Memory overhead** - Must track O(n²) candidate pairs
4. **Linear interactions** - Assumes f1 × f2 form, not f1² or log(f1 × f2)

---

## When to Use This

### Perfect for:
- **Marketing/donor models** - Customer segments interact with offers
- **Healthcare** - Treatments interact with patient characteristics  
- **Recommender systems** - User features interact with item features
- **Financial models** - Market conditions interact with asset types

### Not ideal for:
- **Image/text data** - Better to use neural nets
- **High-dimensional spaces** - Too many pairs to track
- **Non-linear relationships** - Need polynomial features or kernels

---

## Visualization Output

The code produces a **dual-panel plot**:

### Top Panel: Main Feature Weights
- Grey lines: Base features (stable)
- Colored lines: Injected features (tracked individually)
- Green dotted: Feature injection events

### Bottom Panel: Interaction Weights  
- Each line = one interaction term
- Green dashed: Interaction discovery events
- Red X: Pruning events

**What to look for:**
- Interactions that **grow quickly** after discovery (strong signal)
- Interactions that **decay to zero** (false positives, pruned)
- **Correlations** between feature injection and interaction discovery

---

## Tuning Guide

### If interactions are too weak:
```python
interaction_threshold=0.02  # Lower (was 0.03)
grace_period=6000           # Longer incubation
l1_penalty=0.00005          # Weaker gravity
```

### If too many spurious interactions:
```python
interaction_threshold=0.05  # Higher bar
grace_period=3000           # Faster pruning
l1_penalty=0.0002           # Stronger regularization
```

### If missing important interactions:
```python
interaction_search_interval=1000  # Search more often
learning_rate=0.08                # Learn faster
```

---

## Next Steps

### 1. **Run on your data:**
```python
agent, history, events, base, queue = run_simulation(df_final)
visualize_results(agent, history, events, base, queue)
```

### 2. **Validate discoveries:**
Check if found interactions make **domain sense**:
- Does `weather × distance` logically matter?
- Is `age × donation_frequency` a real pattern?

### 3. **A/B test:**
Compare predictions:
- Baseline model (main effects only)
- Interaction model
- Measure lift in AUC/precision

### 4. **Extend to triplets (optional):**
If interactions aren't enough, add 3-way terms:
```python
for (f1, f2, f3) in active_triplets:
    z += w * f1 * f2 * f3
```

---

## Code Architecture

```
InteractionAwareAgent
│
├── Main Effects
│   ├── weights{}          # Feature → weight
│   ├── active_features    # Currently active
│   └── pruned_features    # Removed
│
├── Interactions
│   ├── interaction_weights{}      # (f1,f2) → weight
│   ├── active_interactions        # Active pairs
│   └── candidate_pairs            # Waiting to be tested
│
└── Discovery Mechanism
    ├── residual_correlations{}    # Track correlations
    ├── _update_residual_stats()   # Update after each step
    └── search_interactions()      # Promote strong pairs
```

---

## Real-World Example: Blood Donor Predictions

**Discovered Interactions (Hypothetical):**

1. **weather_suitability × center_distance**  
   Weight: +0.12  
   Interpretation: Bad weather deters distant donors more

2. **age × donation_frequency**  
   Weight: +0.08  
   Interpretation: Older regular donors are MORE reliable (compound loyalty)

3. **is_holiday × reliability_score**  
   Weight: -0.06  
   Interpretation: Holidays disrupt unreliable donors more

4. **transfer_cost × biological_value**  
   Weight: -0.04  
   Interpretation: High-value blood justifies transfer costs

These wouldn't be found by main-effects-only models!

---

## Performance Expectations

On your 100k stream:

- **Runtime:** ~5-8 minutes (vs 3 min for base agent)
- **Memory:** ~2× baseline (stores O(n²) correlations)
- **Accuracy lift:** +2-5% AUC if interactions exist
- **Interpretability:** High - each interaction has clear meaning

---

## Debugging Checklist

If not finding interactions:

- [ ] Check `residual_correlations` - Are any growing?
- [ ] Lower `interaction_threshold` to 0.02
- [ ] Increase `learning_rate` so weights grow faster
- [ ] Verify features are scaled (interactions need comparable magnitudes)
- [ ] Check data - Do interactions actually exist?

If finding too many junk interactions:

- [ ] Raise `interaction_threshold` to 0.05
- [ ] Increase `l1_penalty` for interactions
- [ ] Shorten `grace_period` to prune faster
- [ ] Reduce `interaction_search_interval` to 3000+

---

## Comparison to Your Original Agent

| Feature | Original | Interaction-Aware |
|---------|----------|-------------------|
| Main effects | ✅ Yes | ✅ Yes |
| Interactions | ❌ No | ✅ Yes (pairwise) |
| Discovery | Online | Online |
| Pruning | Grace period | Grace period |
| Complexity | O(n) | O(n²) space |
| Runtime | Fast | ~1.5× slower |
| Interpretability | High | High |

**Verdict:** Use interaction-aware when domain has known multiplicative effects!
