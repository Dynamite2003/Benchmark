# ✅ To-Do List

## 📌 Long-Term Forecasting（长期预测）
- **整理** Long-Term Forecasting（长期预测）结果（MSE/MAE）
- **数据来源**：ETT、Traffic、Weather、ECL（共 16 列）
- **模型对比**（共 11 个模型）：
  - [x] Informer
  - [x] Autoformer
  - [x] TimesNet
  - [X] PatchTST
  - [x] iTransformer
  - [x] Stationary
  - [x] DLinear
  - [x] Timer-XL
  - [x] Crossformer
  - [x] TimeMixer
  - [x] TimeXer

---

## 📌 Zero-Shot Forecasting（零样本预测）
- **整理** Zero-Shot Forecasting 结果（MSE/MAE）
- **数据来源**：ETT、Weather、ECL（共 16 列）
- **模型对比**（共 5 个模型）：
  - [ ] Timer-XL
  - [ ] Time-MoE
  - [ ] Moirai
  - [ ] TimesFM
  - [ ] Chronos
  

---

## 📌 Classification（分类任务）
- **整理** Classification 结果（Accuracy）
- **数据来源**：UEA 数据集（共 11 列）
- **模型对比**（共 4 个模型）：
  - [ ] TimesBERT
  - [ ] Moment
  - [ ] ModernTCN
  - [ ] TimesNet

---

📌 **数据来源**：Timer-XL，TimeXer，TimesBERT 等包含 Baseline Model 的论文  

---

❓** 使用说明与待解决的问题 **

- 现在三个任务的数据均可以进行写入。要求在写数据时保证config.json
中的model名称与父文件夹名称相同

- 目前还不能正确处理对缺少数据的模型的平均值。若缺少数据先写0

- model_type目前只能识别deep-learning，其它类型的在网页上有不影响使用的显示bug

