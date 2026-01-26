# Nursery Irrigation RL（苗圃智能灌溉强化学习）  
**FAO-56 Root-Zone Water Balance + PPO-Optimized（Reward Shaping / UCB / Robust Training）**

本项目面向 **苗圃（nursery）场景的智能灌溉调度**，目标是在 **节水（water-saving）** 的同时 **避免作物水分胁迫（stress）**。  
代码以 **FAO-56** 作为可解释的物理机制（根区亏缺 *Dr*、总有效水量 *TAW*、易用水量 *RAW*、胁迫系数 *Ks*、深层渗漏 *DP*），并在此基础上引入 **PPO 强化学习**，构建可复现的研究型工程结构，支撑 SCI 论文的符号体系、实验表格与图表输出。

---

## 1. 研究目标与核心思想（Paper-friendly）

### 1.1 关键变量（与论文符号一致）
- **根区亏缺**：`Dr`（mm）  
- **体积含水量**：`theta`（m³/m³）  
- **参考蒸散**：`ET0`（mm/day）  
- **作物蒸散**：`ETc`（mm/day）  
- **总有效水量**：`TAW`（mm）  
- **易用水量**：`RAW`（mm）  
- **水分胁迫系数**：`Ks`（dimensionless）  
- **深层渗漏**：`DP`（mm/day）  
- **灌溉动作**：`I`（mm/day）

> 实际部署时，传感器通常直接测到的是 `theta`，但 **FAO-56 的闭环控制核心更适合使用 `Dr`**。  
> 本项目采用 **内部状态 Dr**，并提供 **Dr ↔ theta** 的映射，兼顾理论与工程可落地性。

---

## 2. 快速开始（Quickstart）

### 2.1 安装依赖
```bash
pip install -r requirements.txt
```

### 2.2 训练 PPO（单次）
```bash
python scripts/run_train.py --config configs/train.yaml
```

### 2.3 评估并导出轨迹（trajectory.csv）
```bash
python scripts/run_eval.py --config configs/train.yaml --out outputs/eval_run
```

### 2.4 多随机种子训练（建议 ≥10 seeds）
```bash
python scripts/run_seeds.py --config configs/train.yaml --seeds 10
```

---

## 3. 项目目录结构（模块化降耦合）

```
nursery-irrigation-rl/
  configs/                      # 所有实验配置（论文可复现核心）
  irrigation_rl/                # Python package
    envs/                       # 环境：FAO-56物理模型 + 状态转移 + 天气输入
    rewards/                    # 奖励函数：目标区间 + reward decomposition
    exploration/                # 探索增强：UCB bonus
    robust/                     # 鲁棒训练：domain randomization / adversarial
    baselines/                  # 基线方法：固定阈值/FAO规则
    train/                      # 训练/评估：PPO训练入口 + trajectory导出
    metrics/                    # 指标：MAE/RMSE/Total water/Stress days
  scripts/                      # 一键脚本：train/eval/seeds/figures/tables
  outputs/                      # 实验输出（模型、轨迹csv、表格、图）
```

---

## 4. 各模块说明（你在论文里怎么写，也要在代码里找得到）

### 4.1 `irrigation_rl/envs/`（环境模块）
> 目标：**让 FAO-56 的符号链条在代码里闭环**，并且可单元测试、可复现。

- `nursery_env.py`  
  - `NurseryIrrigationEnv`：基于 `Dr` 的 Gym 环境  
  - observation = `[Dr, theta, ET0, stage_norm]`  
  - action = `I`（灌溉量，mm/day）  
  - ⚠️ 当前版本把 `reward=0` 留给外部注入（便于做 reward ablation）

- `fao56.py`  
  - `calc_TAW / calc_RAW / calc_Ks`：FAO-56关键公式  
  - `theta_to_Dr / Dr_to_theta`：状态映射  
  - `calc_ET0_PM`：**Penman–Monteith（目前为占位，内部 fallback）**
  - `calc_ET0_fallback`：传感器受限的简化 ET0（可替换 Hargreaves）

- `dynamics.py`  
  - `update_Dr()`：根区水量平衡更新（最核心、最应该与论文公式一致）

- `weather.py`  
  - `AssumptionWeatherProvider`：缺失变量用固定值/噪声补齐  
  - `ExternalCSVWeatherProvider`：支持从 CSV 读取天气（推荐用于论文实验）

✅ **论文对应（建议写法）**：  
Section 3.4 “Transition Dynamics” 可逐行对应 `update_Dr()`；  
Table 1 的 `TAW/RAW/Ks/Dr/ET0/ETc` 可逐条对应 `fao56.py` 中函数。

---

### 4.2 `irrigation_rl/rewards/`（奖励模块）
> 目标：奖励设计必须能“写成公式 + 可消融 + 可复现”。

- `target.py`  
  - `DynamicTarget.get_interval()`：返回 `Dr` 的目标区间 `[Dr_lo, Dr_hi]`  
  - 当前实现为：  
    - `Dr_lo = low_frac_of_TAW * TAW`  
    - `Dr_hi = high_frac_of_RAW * RAW`

- `reward.py`  
  - `RewardFunction.compute()`：将 reward 分解为：  
    - `r_track`：对区间误差惩罚（distance-to-interval）  
    - `r_water`：灌溉量惩罚（节水）  
    - `r_improve`：误差下降的“进步奖励”  
    - `r_smooth`：动作平滑惩罚  
    - `r_safe`：安全/极端状态惩罚  
    - `r_ucb`：探索 bonus（可开关）

✅ **论文对应**：  
Section 4.3 “Reward Shaping” 可以完全对齐 `RewardConfig` 和 `terms`。

---

### 4.3 `irrigation_rl/exploration/`（UCB探索）
> 目标：让论文里的 UCB 不再是口头描述，而是可复现实现。

- `ucb_bonus.py`  
  - `ActionBinUCB`：将连续动作离散为 bins，并维护计数 `counts`  
  - `bonus(t, bin)` = `c * sqrt(log(t+1)/(n+1))`

✅ **论文对应**：  
Section 4.4 “UCB Exploration” 直接对应这个模块。

---

### 4.4 `irrigation_rl/robust/`（鲁棒训练）
> 目标：鲁棒性必须发生在 **训练阶段（train-time）**，不是只做扰动测试。

- `randomization.py`  
  - `apply_domain_randomization()`：生成 `ET0_mult / Kc_mult / Zr_mult / theta_sigma` 等扰动参数  
  - 当前为骨架（后续我们会将其真正注入 env/reset）

- `adversarial.py`  
  - `choose_worst_delta()`：有限扰动集合选最坏（可选，论文可写成可扩展）

✅ **论文对应**：  
Section 4.5 “Disturbance-Robust Training”。

---

### 4.5 `irrigation_rl/train/`（训练与评估）
- `ppo_train.py`  
  - `train_ppo()`：SB3 PPO 训练入口  
  - `build_env()`：从 `env.yaml` 构建环境与天气源

- `evaluate.py`  
  - `evaluate_policy()`：加载模型并导出 `trajectory.csv`  
  - 目前在 evaluate 内部计算 reward（便于迅速出图/出指标）

---

### 4.6 `irrigation_rl/metrics/`（指标模块）
- `metrics.py`  
  - `compute_metrics()`：输出 MAE/RMSE/TotalIrrigation/StressDays  
  - 可扩展：Termination rate、Worst-case MAE、ΔMAE 等

---

## 5. 配置文件说明（configs/*.yaml）

> ✅建议：论文里的 Table 2/3/6/7 对应配置文件字段，做到“写论文=抄配置”。

---

### 5.1 `configs/env.yaml`（环境与物理参数）
#### `scenario`
- `horizon_days`：仿真时长（天），论文常用 `H=90`
- `a_max_mm`：动作上限（mm/day），例如 `15.0`
- `dt_days`：时间步长（天），当前为 1 天一步

#### `soil`
- `theta_fc`：田间持水量（Field Capacity, m³/m³）
- `theta_wp`：凋萎点（Wilting Point, m³/m³）
- `Zr_m`：根区深度（m）
- `p`：可提取水比例（RAW 系数）

#### `crop`
- `Kc_ini / Kc_mid / Kc_end`：作物系数阶段值
- `stage_ini_days / stage_mid_days / stage_end_days`：各阶段天数（总和建议等于 horizon_days）

#### `weather`
- `mode`：天气输入模式  
  - `assumption`：固定值（适合跑通骨架）  
  - `external`：从 CSV 加载（论文建议）  
  - `measured`：对接真实传感数据（后续扩展）
- `csv_path`：external 模式的 CSV 文件路径  
- `T_mean_C / RH_pct / u2_mps / Rs_MJ_m2_day`：assumption 模式默认天气值

#### `termination`
- `terminate_on_theta_below_wp`：若 `theta < theta_wp` 是否终止
- `terminate_on_Dr_above_TAW`：若 `Dr >= TAW` 是否终止

---

### 5.2 `configs/train.yaml`（训练主配置）
- `seed`：随机种子
- `total_timesteps`：训练总步数
- `policy`：SB3 policy 类型（默认 `MlpPolicy`）

#### `ppo`
- `learning_rate`：学习率
- `n_steps`：rollout buffer step（影响更新频率）
- `batch_size`：mini-batch 大小
- `n_epochs`：每次更新的 epoch 数
- `gamma`：折扣因子（论文符号通常为 `γ`）
- `gae_lambda`：GAE lambda（论文符号通常为 `λ`）
- `clip_range`：PPO clip epsilon（论文符号通常为 `ε`）
- `ent_coef`：熵正则系数（促进探索）

#### `ablation`
用于消融实验（后续可在训练入口根据开关注入模块）
- `use_dynamic_target`
- `use_reward_shaping`
- `use_ucb_bonus`
- `use_robust_training`

#### `paths`
- `env_config`：指向 `env.yaml`
- `out_dir`：输出目录（建议永远写到 `outputs/`）

---

### 5.3 `configs/noise_train.yaml`（训练扰动）
- `enabled`：是否启用
- `weather_bias.ET0_mult_min/max`：ET0 乘性扰动范围
- `sensor_noise.theta_sigma`：观测噪声（theta）
- `param_noise.Kc_mult_* / Zr_mult_*`：作物参数/根深扰动

✅ 论文写法：train-time domain randomization（弱扰动）

---

### 5.4 `configs/noise_test.yaml`（测试扰动）
与 train 类似，但一般范围更宽：  
✅ 论文写法：domain shift / robustness evaluation（强扰动）

---

### 5.5 `configs/ablation.yaml`（消融开关）
用于快速指定消融组合。  
后续可以在 `run_eval.py` / `run_seeds.py` 加入读取该文件以自动跑 Table 10。

---

## 6. 输出说明（outputs/）

建议输出结构（后续完善为论文友好格式）：
```
outputs/
  ppo_seed42.zip
  eval_run/
    trajectory.csv
  seed_XX/
    model.zip
    trajectory.csv
    metrics.json
```

- `trajectory.csv`：用于画 Fig.7（Dr轨迹/目标区间/灌溉量）  
- `metrics.json`：用于生成 Table 8/9/10（mean±std）

---

## 7. 如何对齐论文图表（Fig/Table Mapping）

> 下面是建议的“论文-代码”映射方式（后续我们将补齐脚本）。

- **Fig.6**：训练曲线（Return/MAE vs steps）  
  - 来源：SB3 logs（可扩展 TensorBoard）

- **Fig.7**：`Dr`（或 `theta`）随时间轨迹 + 目标区间  
  - 来源：`outputs/*/trajectory.csv`

- **Fig.8**：总灌溉量对比（bar chart）  
  - 来源：`compute_metrics()` 的 `TotalIrrigation`

- **Fig.9**：扰动 vs 标称鲁棒性对比  
  - 来源：`noise_test.yaml` 下的评估输出

- **Table 8/9/10**：主结果/扰动结果/消融结果  
  - 来源：多 seed 汇总 mean±std（由 `scripts/make_tables.py` 聚合）

---

## 8. 当前骨架的“已实现 / 待实现”清单（非常重要）

### ✅ 已实现（可跑通）
- Dr-based Gym env（`NurseryIrrigationEnv`）
- FAO-56 基本符号函数：TAW/RAW/Ks、theta↔Dr
- PPO 训练入口（SB3）
- trajectory 导出
- UCB bonus（计数 + bonus）

### 🟡 待实现（我们下一步要补齐）
- `calc_ET0_PM()`：完整 Penman–Monteith（支持 sensor-limited fallback）
- `DP` 深层渗漏（建议从 `theta > theta_fc` 推出）
- train-time domain randomization 真正注入 env/reset
- 多 seeds 结果聚合（Table 8/9/10）与作图脚本（Fig.6–11）
- ablation 与 robust 的开关体系在训练/评估中的真正生效

---

## 9. 常见问题（FAQ）

### Q1：传感器不足，Penman–Monteith 怎么办？
A：本项目允许三种策略（论文里要写清楚）：
1) external weather（推荐）
2) greenhouse assumption（固定风速/辐射等）
3) fallback approximation + sensitivity analysis（证明结论稳健）

### Q2：为什么内部用 Dr 而不是直接 θ？
A：`theta` 是观测量（传感器读数），`Dr` 是 FAO-56 的控制量，物理解释更清楚，论文更可信。

### Q3：为什么 reward 不写在 env.step 里？
A：为支持消融实验（w/o reward shaping）与不同 reward 组合，本项目把 reward 独立成模块，便于复现与对照实验。

---

## 10. 下一步开发路线（建议顺序）
1) **补齐 ET0_PM**（核心可解释性）
2) **补齐 DP 模型**（水量闭环）
3) **把 randomization 注入 reset**（鲁棒训练生效）
4) **写 scripts/make_tables.py & make_figures.py**（论文自动生成）
5) **跑 10 seeds + 输出 Table 8/9/10**（SCI 可复现）

---

如需我继续推进：  
✅ 我可以下一步直接给你 **`calc_ET0_PM()` 的最小可运行 Penman–Monteith 实现**（支持缺失变量补齐 & 单元测试）。  
你只需回复：**“开始实现 ET0_PM”**。
