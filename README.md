# text2preset

本專案目標是建立一套以文字描述產生音訊效果參數的工作流程，方便音訊後製與創作團隊快速取得合適的 preset。下方整理現況、開發指引與資料集資訊，提供團隊對齊與新加入成員 onboarding。

## 專案現況

- 進度追蹤集中於 issue [#17](https://github.com/vaclisinc/text2preset/issues/17)，建議開發前先瀏覽。
- `baseline-system/` 已串起完整 refine loop：LLM 產生參數 → 套用 → CLAP 打分 → 依分數再提示。
- 目前尚未接上 plugin chain 的實際套用流程，評分直接使用 `audio_samples/` 中尚未處理的音檔。補上該步驟後即完成第一版 baseline。

## 快速開始

1. 建立 Python 虛擬環境（若尚未準備）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安裝相依套件：
   ```bash
   cd baseline-system
   pip install -r requirements.txt
   ```
3. 進行 smoke test（觀察完整 refine loop 流程）：
   ```bash
   pytest tests/test_refine_loop.py -s
   ```

## Baseline System 設定

- `configs/default.yaml`：設定使用的模型、API endpoint、chain 參數等。需要多組設定時可在 `configs/` 下新增 YAML。
- `prompts/`：包含 generation、judge、refine 三份 system prompt。若 loop 行為異常，先檢查與調整這些 prompt。
- `src/`：主要程式碼（refine loop、LLM wrapper、評分器、工具函式等）。
- `tests/`：單元與整合測試。`tests/test_refine_loop.py` 與 `tests/test_generation.py` 是目前最能呈現流程的案例。
- `audio_samples/`：現階段用於評分的原始音檔。等 plugin chain 套用實作完成後，評分流程會改為處理後音訊。

## 目錄導覽

- `baseline-system/`：baseline pipeline 原始碼與測試。
- `ref/`：研究資料與參考資料集。
- `CLAUDE.md`：與 Claude 協作的 TDD 流程與注意事項。
- `1016-LLM-as-music-judge (1).pdf`：相關研究文獻。

## Dataset 說明

`ref/` 目錄目前包含兩筆主要資料：

- `ref/social-data/`：SocialFX 的原始資料，細節請參考資料夾內的 `SocialFX_paper.pdf`。
- `ref/fx-processor/`：Sony 2024 論文 *Can Large Language Models Predict Audio Effects Parameters from Natural Language?* 所使用的處理後乾淨資料。

這些資料尚未整合進 baseline pipeline，但提供了參考案例與後續擴充素材。

## 後續建議

1. 補上 plugin chain 套用流程，讓評分來源改為處理後的音訊。
2. 視需要擴增 `tests/` 下的 integration 測試，確保 chain 執行穩定。
3. 研究如何將 `ref/` 中的資料集轉換為適合 baseline 系統的訓練或 prompt 素材。
