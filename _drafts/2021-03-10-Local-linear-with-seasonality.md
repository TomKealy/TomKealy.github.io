---
layout: default
title:  "Local Linear Trend models with seasonality"
date:   2021-03-10
categories: time-series
---


To solve this, we can add a seasonal component:

$$ u_t \sim N\left(u_{t-1}, \sigma_v^2\right) $$

$$ s_t \sim N\left(-\sum^n_{l=1}s_{t-l}, \sigma_s\right) $$

$$ y_t \sim N\left(u_t + s_t, \sigma_y^2\right) $$


```python
stan_code = """data {
    int N;
    int pred_num;
    vector[N] y;
}

parameters {
    vector[N] s;
    vector[N] u;
    real<lower=0> s_s;
    real<lower=0> s_u;
    real<lower=0> s_y;
}

model {
    s[12:N] ~ normal(-s[1:N-11]-s[2:N-10]-s[3:N-9]-s[4:N-8]-s[5:N-7]-s[6:N-6]-s[7:N-5]-s[8:N-4]-s[9:N-3]-s[10:N-2]-s[11:N-1], s_s);
    u[2:N] ~ normal(u[1:N-1], s_u);
    y ~ normal(u+s, s_y);
}

generated quantities {
    vector[N+pred_num] s_pred;
    vector[N+pred_num] u_pred;
    vector[N+pred_num] y_pred;

    s_pred[1:N] = s;
    u_pred[1:N] = u;
    y_pred[1:N] = y;

    for (t in (N+1):(N+pred_num)){
        s_pred[t] = normal_rng(-s_pred[t-11]-s_pred[t-10]-s_pred[t-9]-s_pred[t-8]-s_pred[t-7]-s_pred[t-6]-s_pred[t-5]-s_pred[t-4]-s_pred[t-3]-s_pred[t-2]-s_pred[t-1], s_s);
        u_pred[t] = normal_rng(u_pred[t-1], s_u);
        y_pred[t] = normal_rng(u_pred[t]+s_pred[t], s_y);
    }
}
"""
```
By running this model and plotting using the code above we get a much better fit!