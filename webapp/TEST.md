# 測試指南

## 當前狀態

✅ **已完成（Phase 1）：**
- 後端 API 正常運作
- 前端 UI 已修復錯誤
- API keys 已配置（3個都有）
- 可以生成參數

❌ **未完成（Phase 2）：**
- 音訊實際處理
- 音訊播放

## 快速測試步驟

### 1. 啟動應用

**方法 1：兩個終端機**
```bash
# Terminal 1 - 後端
cd /Users/vaclis./Documents/UCB/CNMAT/text2preset/webapp/backend
source venv/bin/activate
python main.py

# Terminal 2 - 前端
cd /Users/vaclis./Documents/UCB/CNMAT/text2preset/webapp/frontend
npm run dev
```

### 2. 打開瀏覽器
訪問：http://localhost:5173

### 3. 測試流程

1. **檢查模型載入**
   - 應該看到 LLM Model 下拉選單
   - 有 OpenRouter、OpenAI、Claude 選項
   - 如果看到紅色錯誤，重新整理頁面

2. **測試簡單提示詞**
   ```
   warm and spacious
   ```
   - 點擊 "Generate Parameters"
   - 等待 2-5 秒
   - 應該看到 Reverb、EQ、Compressor 參數

3. **測試你的範例**
   ```
   after rain campus in October
   ```
   - 切換不同 LLM 模型看效果差異
   - 查看不同參數分頁

4. **測試複雜提示詞**
   ```
   large cathedral with long reverb tail and bright treble boost
   ```
   - 展開 "View Full JSON" 查看完整參數

## 關於音訊處理

### 為什麼沒做？

Phase 1 的目標是**驗證核心概念**：
- 文字能否有效轉換成參數？✅
- LLM 能否理解音效描述？✅
- 參數格式是否正確？✅

### 如何展示音訊效果？

**選項 1：展示參數 + 口頭說明（推薦）**
- 展示生成的參數
- 說明這些參數的意義
- 說明 Phase 2 會實際套用

**選項 2：離線展示（如果時間充足）**
- 使用現有的 `ref/fx-processor` 離線處理
- 準備一些預先處理好的音訊樣本
- 展示前後對比

**選項 3：快速加入音訊處理（需要 2-3 小時）**
- 整合 fx-processor 到 webapp
- 加入音訊播放器
- 風險：可能影響明天 demo 的穩定性

## Demo 建議

### 開場（1分鐘）
"這是一個 LLM-powered 的音效參數生成系統。使用者只需要用自然語言描述想要的音效，系統就能自動生成專業的 Reverb、EQ、Compressor 參數。"

### 展示（3分鐘）
1. **簡單範例**："warm and spacious" → 展示參數
2. **你的範例**："after rain campus in October" → 切換不同 LLM
3. **複雜範例**："cathedral reverb with bright EQ" → 展開 JSON

### 技術細節（1分鐘）
- 多 LLM 支援（OpenRouter、Claude、GPT-4）
- 參數格式符合 SocialFX 標準
- JSON 可導出給其他工具使用

### 未來展望（1分鐘）
"Phase 2 會加入實際音訊處理，使用 Web Audio API 直接在瀏覽器中套用這些參數，並提供 A/B 比較播放功能。"

## 如果一定要加音訊處理

需要決定：

### A. 使用現有 fx-processor（較快）
- 優點：代碼已存在，只需整合
- 缺點：需要 Node.js + Chrome，部署複雜
- 時間：2-3 小時

### B. 純前端 Web Audio（較慢）
- 優點：不需要後端處理，部署簡單
- 缺點：需要重新實作音效算法
- 時間：4-5 小時

### C. 只展示參數（最推薦）
- 優點：穩定、快速、專注核心
- 缺點：沒有實際音訊輸出
- 時間：0 小時（已完成）

## 當前錯誤已修復

剛才的錯誤 "Cannot read properties of null" 已經修復。原因是：
- 後端 `/api/models` 返回空的 models
- 前端沒有檢查就直接存取
- 已加入錯誤處理和友善提示

現在如果沒有 API key，會顯示清楚的錯誤訊息。

## 測試清單

在明天 demo 之前：

- [ ] 啟動後端，確認無錯誤
- [ ] 啟動前端，確認無錯誤
- [ ] 測試至少 3 個不同的提示詞
- [ ] 測試切換不同 LLM 模型
- [ ] 準備好範例提示詞列表
- [ ] 檢查 API key 額度是否充足
- [ ] 準備備用計畫（如果 API 失敗）

## 緊急備案

如果 demo 時出問題：

1. **API 失敗**：展示預先截圖
2. **網路問題**：使用本地 mock 數據
3. **瀏覽器問題**：切換到 Chrome/Firefox
4. **後端崩潰**：重啟後端（不到 10 秒）

## 結論

**建議：明天只展示 Phase 1**
- 核心功能完整且穩定
- 專注展示 LLM 參數生成
- 說明 Phase 2 的規劃即可
- 避免最後一刻加新功能的風險

如果你堅持要加音訊處理，我可以幫你，但需要確認：
1. 你有多少時間？
2. 願意承擔不穩定的風險嗎？
3. 音訊處理對 demo 的重要性？
