import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from wireless_optim.environment import HetNetEnvironment

@dataclass
class AblationResult:
    name: str
    avg_power_per_bs: np.ndarray
    avg_bandwidth_per_user: np.ndarray        
    avg_pf_score_per_user: np.ndarray         
    avg_sum_rate_mbps: float                  
    jain_fairness: float                    

    
    avg_power_per_bs_norm: np.ndarray         
    avg_bandwidth_per_user_norm: np.ndarray   
    avg_sched_score_per_user_norm: np.ndarray

def _safe_log2(x): return jnp.log(x) / jnp.log(2.0)

def _geom_gain_proxy(env, state):
    bs_pos = state['bs_positions']              
    ue_pos = state['user_positions']            
    d = jnp.linalg.norm(bs_pos[:, None] - ue_pos[None, :], axis=-1)  
    pl_db = 10.0 * env.path_loss_exponent * jnp.log10(d + 1e-3)
    pl_lin = 10 ** (pl_db / 10.0)
    g = 1.0 / pl_lin                             
    return g, d

def _initial_pf_memory(num_users):
    return jnp.ones((num_users,)) * 1.0 

def _pf_update(avg_rate, inst_rate, alpha=0.01):
    return (1 - alpha) * avg_rate + alpha * inst_rate

def _build_action(env, state, power_levels, bw_frac_per_bs, sched_scores):
    curr_p = state['power_levels']
    delta_db = jnp.clip(power_levels - curr_p, -5.0, 5.0)   
    power_action = (delta_db / 10.0) + 0.5                  

    bw_action = jnp.clip(bw_frac_per_bs, 0.0, 1.0)
    sched_flat = sched_scores.reshape(-1)
    sched_norm = sched_flat / (jnp.std(sched_flat) + 1e-6)  

    return jnp.concatenate([power_action, bw_action, sched_norm]).astype(jnp.float32)

def _metrics_from_step(env, ts, state_before, state_after):
    bs_pos = state_after['bs_positions']
    ue_pos = state_after['user_positions']
    d = jnp.linalg.norm(bs_pos[:, None] - ue_pos[None, :], axis=-1)
    pl_db = 10.0 * env.path_loss_exponent * jnp.log10(d + 1e-3)
    pl_lin = 10 ** (pl_db / 10.0)

    p_lin_mw = 10 ** (state_after['power_levels'] / 10.0)          
    rx = p_lin_mw[:, None] / pl_lin                                 
    serving = jnp.argmax(rx, axis=0)                                 
    sig = jnp.max(rx, axis=0)                                        
    interf = jnp.sum(rx, axis=0) - sig                               
    noise_lin = 10 ** (env.noise_floor_dbm / 10.0)
    sinr = sig / (interf + noise_lin + 1e-9)
    rate_per_hz = _safe_log2(1.0 + sinr)                             
    alloc = state_after['resource_allocations']                      
    one_hot = jax.nn.one_hot(serving, env.num_bs, dtype=alloc.dtype) 
    bw_for_user = jnp.sum(alloc.T * one_hot, axis=1)                 
    thr = bw_for_user * rate_per_hz                                  

    sum_rate_mbps = jnp.sum(thr) / 1e6
    jain = (jnp.sum(thr) ** 2) / (thr.shape[0] * jnp.sum(thr ** 2) + 1e-12)

    return np.array(bw_for_user), np.array(rate_per_hz), float(sum_rate_mbps), float(jain)


def policy_pf_equal(env, state, pf_mem):
    """
    PF-Equal (PF-EQ): each BS uses full bandwidth; per-BS softmax over PF scores.
    Power: mild load-aware adjustment around current levels.
    """
    g, _ = _geom_gain_proxy(env, state)               
    best_bs = jnp.argmax(g, axis=0)
    best_g = jnp.max(g, axis=0)
    inst_rate_proxy = _safe_log2(1 + best_g)          
    pf_score = inst_rate_proxy / (pf_mem + 1e-6)     
    mask_BU = jax.nn.one_hot(best_bs, env.num_bs, dtype=bool).T   
    sched_scores = jnp.where(mask_BU, pf_score[None, :], 0.5 * pf_score[None, :])  
    bw_frac = jnp.ones((env.num_bs,))
    load_per_bs = jnp.sum(jnp.argmax(g, axis=0)[:, None] == jnp.arange(env.num_bs), axis=0)
    load_norm = load_per_bs / (jnp.max(load_per_bs) + 1e-6)
    target_p = jnp.clip(state['power_levels'] + 2.0 * (load_norm - 0.5), 0.0, env.max_power)
    action = _build_action(env, state, target_p, bw_frac, sched_scores)
    return action, pf_score, sched_scores

def policy_greedy_ofdma(env, state, topk=4):
    g, _ = _geom_gain_proxy(env, state)               
    def bs_scores(row):
        idx = jnp.argsort(-row)[:topk]
        s = jnp.zeros_like(row)
        s = s.at[idx].set(row[idx] / (jnp.max(row[idx]) + 1e-9))
        return s

    sched_scores = jax.vmap(bs_scores)(g)             
    strength = jnp.mean(jnp.sort(g, axis=1)[:, :topk], axis=1)
    bw_frac = strength / (jnp.max(strength) + 1e-9)
    bw_frac = jnp.clip(bw_frac, 0.2, 1.0)
    med_g = jnp.median(g, axis=1)
    bump = jnp.clip(1.5 - med_g / (jnp.max(med_g) + 1e-9), 0.0, 1.0) * 3.0
    target_p = jnp.clip(state['power_levels'] + bump, 0.0, env.max_power)

    action = _build_action(env, state, target_p, bw_frac, sched_scores)
    best_g = jnp.max(g, axis=0)
    inst_rate_proxy = _safe_log2(1 + best_g)
    pf_score = inst_rate_proxy / 1.0
    return action, pf_score, sched_scores

def policy_interference_pricing_power(env, state, gamma_db=5.0):
    g, _ = _geom_gain_proxy(env, state)               
    best_bs = jnp.argmax(g, axis=0)                   
    g_ii = g[best_bs, jnp.arange(env.num_users)]      
    sorted_g = jnp.sort(g, axis=0)[-3:-1, :]          
    I_proxy = jnp.sum(sorted_g, axis=0) + 10 ** (env.noise_floor_dbm / 10.0)
    gamma = 10 ** (gamma_db / 10.0)
    p_req_user_mw = gamma * I_proxy / (g_ii + 1e-12)
    p_req_user_dbm = 10 * jnp.log10(p_req_user_mw + 1e-12)  
    bs_ids = jnp.arange(env.num_bs)[:, None]                         
    mask_BU = (best_bs[None, :] == bs_ids)                           
    counts = jnp.sum(mask_BU, axis=1)                                
    sum_p = jnp.sum(jnp.where(mask_BU, p_req_user_dbm[None, :], 0.0), axis=1)  
    avg_p_req = jnp.where(counts > 0, sum_p / counts, state['power_levels'])  
    target_p = jnp.clip(avg_p_req, 0.0, env.max_power)
    pf_like = _safe_log2(1 + g_ii)                                   
    mask_sched = jax.nn.one_hot(best_bs, env.num_bs, dtype=bool).T   
    sched_scores = jnp.where(mask_sched, pf_like[None, :], 0.2 * pf_like[None, :])
    bw_frac = jnp.ones((env.num_bs,))
    action = _build_action(env, state, target_p, bw_frac, sched_scores)
    return action, pf_like, sched_scores

def run_baseline(env, key, policy_fn, steps=100):
    ts = env.reset(key)
    state = env._state
    num_bs, num_users = env.num_bs, env.num_users

    pf_mem = _initial_pf_memory(num_users)
    acc_power_dbm = []
    acc_power_norm = []
    acc_bw_hz = []
    acc_bw_norm = []
    acc_sched_norm = []
    acc_pf = []
    acc_sumrate = []
    acc_jain = []

    p_max_lin_mw = 10 ** (env.max_power / 10.0) 

    for _ in range(steps):
        if policy_fn.__name__ == "policy_pf_equal":
            action, pf_score, sched_scores = policy_pf_equal(env, state, pf_mem)
        else:
            action, pf_score, sched_scores = policy_fn(env, state)

        ts = env.step(action)
        new_state = env._state
        bs_pos = new_state['bs_positions']; ue_pos = new_state['user_positions']
        d = jnp.linalg.norm(bs_pos[:, None] - ue_pos[None, :], axis=-1)
        pl_db = 10.0 * env.path_loss_exponent * jnp.log10(d + 1e-3)
        pl_lin = 10 ** (pl_db / 10.0)

        p_lin_mw = 10 ** (new_state['power_levels'] / 10.0)        
        rx = p_lin_mw[:, None] / pl_lin
        serving = jnp.argmax(rx, axis=0)                            
        sig = jnp.max(rx, axis=0)                                   
        interf = jnp.sum(rx, axis=0) - sig                          
        noise_lin = 10 ** (env.noise_floor_dbm / 10.0)
        sinr = sig / (interf + noise_lin + 1e-9)
        rate_per_hz = _safe_log2(1.0 + sinr)                        

        alloc = new_state['resource_allocations']                    
        one_hot = jax.nn.one_hot(serving, env.num_bs, dtype=alloc.dtype)  
        bw_user_hz = jnp.sum(alloc.T * one_hot, axis=1)           

        power_norm = (p_lin_mw / p_max_lin_mw)                      
        bw_norm = bw_user_hz / env.total_bandwidth                  
        sched_row = sched_scores[serving, :]                        
        sched_prob = jax.nn.softmax(sched_row, axis=1)[jnp.arange(num_users), jnp.arange(num_users)] 
        sum_rate_mbps = jnp.sum(bw_user_hz * rate_per_hz) / 1e6
        jain = (jnp.sum(bw_user_hz * rate_per_hz) ** 2) / (num_users * jnp.sum((bw_user_hz * rate_per_hz) ** 2) + 1e-12)
        pf_mem = _pf_update(pf_mem, rate_per_hz)
        acc_power_dbm.append(np.array(new_state['power_levels']))
        acc_power_norm.append(np.array(power_norm))
        acc_bw_hz.append(np.array(bw_user_hz))
        acc_bw_norm.append(np.array(bw_norm))
        acc_sched_norm.append(np.array(sched_prob))
        acc_pf.append(np.array(pf_score))
        acc_sumrate.append(float(sum_rate_mbps))
        acc_jain.append(float(jain))

        state = new_state
        if ts.last():
            break

    avg_power_dbm = np.mean(np.stack(acc_power_dbm, 0), axis=0)
    avg_power_norm = np.mean(np.stack(acc_power_norm, 0), axis=0)
    avg_bw_hz = np.mean(np.stack(acc_bw_hz, 0), axis=0)
    avg_bw_norm = np.mean(np.stack(acc_bw_norm, 0), axis=0)
    avg_sched_norm = np.mean(np.stack(acc_sched_norm, 0), axis=0)
    avg_pf = np.mean(np.stack(acc_pf, 0), axis=0)
    avg_sumrate = float(np.mean(acc_sumrate))
    avg_jain = float(np.mean(acc_jain))

    return (avg_power_dbm, avg_power_norm,
            avg_bw_hz, avg_bw_norm,
            avg_sched_norm, avg_pf,
            avg_sumrate, avg_jain)

def evaluate_all_baselines(env, seed=0, steps=None):
    key = jax.random.PRNGKey(seed)
    steps = steps or env.max_steps
    results = []
    policy_list = [
        ("PF-EQ",   policy_pf_equal),
        ("G-OFDMA", policy_greedy_ofdma),
        ("IP-PC",   policy_interference_pricing_power),
    ]

    for name, pol in policy_list:
        (avg_power_dbm, avg_power_norm,
         avg_bw_hz, avg_bw_norm,
         avg_sched_norm, avg_pf,
         avg_sumrate, avg_jain) = run_baseline(env, key, pol, steps=steps)

        results.append(AblationResult(
            name=name,
            avg_power_per_bs=avg_power_dbm,
            avg_bandwidth_per_user=avg_bw_hz,
            avg_pf_score_per_user=avg_pf,
            avg_sum_rate_mbps=avg_sumrate,
            jain_fairness=avg_jain,
            avg_power_per_bs_norm=avg_power_norm,
            avg_bandwidth_per_user_norm=avg_bw_norm,
            avg_sched_score_per_user_norm=avg_sched_norm,
        ))

    return results

SEEDS = [0, 10, 18, 28, 42, 64, 128, 256, 512, 1024]

def make_env():
    return HetNetEnvironment(num_macro_bs=3, num_small_bs=10, num_users=50, max_steps=100)

def evaluate_baselines_over_seeds(seeds=SEEDS, steps=None, outdir="ablation_out_norm"):
    os.makedirs(outdir, exist_ok=True)

    acc = {
        "PF-EQ":   {"p": [], "bw": [], "sched": []},
        "G-OFDMA": {"p": [], "bw": [], "sched": []},
        "IP-PC":   {"p": [], "bw": [], "sched": []},
    }

    for s in seeds:
        env = make_env()
        res_list = evaluate_all_baselines(env, seed=s, steps=steps or env.max_steps)
        for r in res_list:
            acc[r.name]["p"].append(np.asarray(r.avg_power_per_bs_norm))          
            acc[r.name]["bw"].append(np.asarray(r.avg_bandwidth_per_user_norm))   
            acc[r.name]["sched"].append(np.asarray(r.avg_sched_score_per_user_norm))  

    summary_rows = []
    out = {}
    for name, d in acc.items():
        P   = np.stack(d["p"], axis=0)    
        BW  = np.stack(d["bw"], axis=0)    
        SCH = np.stack(d["sched"], axis=0) 
        p_mean_per_bs   = P.mean(axis=0)       
        bw_mean_per_ue  = BW.mean(axis=0)      
        sch_mean_per_ue = SCH.mean(axis=0)     
        np.savetxt(os.path.join(outdir, f"{name}_power_norm_mean_per_bs.csv"),
                   p_mean_per_bs, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{name}_bandwidth_norm_mean_per_user.csv"),
                   bw_mean_per_ue, delimiter=",")
        np.savetxt(os.path.join(outdir, f"{name}_sched_norm_mean_per_user.csv"),
                   sch_mean_per_ue, delimiter=",")

        p_overall_mean  = P.mean()     
        p_overall_std   = P.std()
        bw_overall_mean = BW.mean()    
        bw_overall_std  = BW.std()
        sch_overall_mean= SCH.mean()   
        sch_overall_std = SCH.std()

        summary_rows.append({
            "method": name,
            "power_norm_mean": p_overall_mean,
            "power_norm_std":  p_overall_std,
            "bw_norm_mean":    bw_overall_mean,
            "bw_norm_std":     bw_overall_std,
            "sched_norm_mean": sch_overall_mean,
            "sched_norm_std":  sch_overall_std,
            "num_seeds":       P.shape[0],
        })

        out[name] = {
            "power_norm_mean_per_bs":   p_mean_per_bs,
            "bandwidth_norm_mean_per_user": bw_mean_per_ue,
            "sched_norm_mean_per_user": sch_mean_per_ue,
            "power_norm_tensor":   P,   
            "bandwidth_norm_tensor":BW,  
            "sched_norm_tensor":   SCH,  
        }

    df_summary = pd.DataFrame(summary_rows).sort_values("method")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df_summary.to_csv(os.path.join(outdir, "normalized_summary_over_seeds.csv"), index=False)
    return out, df_summary
norm_details, norm_summary = evaluate_baselines_over_seeds(
    seeds=[0, 10, 18, 28, 42, 64, 128, 256, 512, 1024],
    steps=1000,
    outdir="ablation_out_norm"
)
