# Text2Preset Web App - Quick Start Guide

快速啟動指南，明天 demo 用！

## 前置準備

### 1. 安裝 API Key（至少一個）

編輯 `baseline-system/.env`：

```bash
# 至少設定一個 API key
OPENROUTER_API_KEY=sk-or-...        # 推薦：便宜快速
OPENAI_API_KEY=sk-...                # 或使用 OpenAI
ANTHROPIC_API_KEY=sk-ant-...         # 或使用 Claude
```

### 2. 確認環境

```bash
python3 --version  # 需要 3.9+
node --version     # 需要 18+
```

## 快速啟動（3 步驟）

### 方法 1：使用啟動腳本（推薦）

```bash
cd /Users/vaclis./Documents/UCB/CNMAT/text2preset/webapp
chmod +x start.sh
./start.sh
```

### 方法 2：手動啟動（兩個終端機）

**Terminal 1 - 後端：**
```bash
cd webapp/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Terminal 2 - 前端：**
```bash
cd webapp/frontend
npm install
npm run dev
```

## 訪問應用

打開瀏覽器：
- **前端界面**：http://localhost:5173
- **API 文檔**：http://localhost:8000/docs
- **健康檢查**：http://localhost:8000/api/health

## Demo 流程

1. **輸入文字描述**
   - 範例：`"warm cathedral reverb with bright treble boost"`
   - 範例：`"tight punchy compression with boosted bass"`
   - 範例：`"spacious hall ambience with balanced frequency"`

2. **選擇 LLM 模型**
   - OpenRouter Claude Haiku：最快最便宜（推薦 demo 用）
   - Claude Sonnet：效果較好但較貴
   - GPT-4：效果好但最貴

3. **（可選）上傳音訊檔案**
   - 支援格式：WAV, MP3, FLAC, OGG
   - Phase 1 只會儲存，不會處理

4. **點擊 Generate Parameters**
   - 等待 2-5 秒（根據模型速度）
   - 查看生成的參數

5. **查看結果**
   - 切換 Reverb / EQ / Compressor 分頁
   - 展開 "View Full JSON" 查看完整參數
   - 可複製 JSON 供其他工具使用

## 疑難排解

### 後端無法啟動
```bash
cd webapp/backend
source venv/bin/activate
python main.py
# 查看錯誤訊息
```

常見問題：
- **缺少 API key**：編輯 `baseline-system/.env`
- **缺少模組**：`pip install -r requirements.txt`
- **路徑錯誤**：確認在 `webapp/backend/` 目錄中

### 前端無法連接後端
```bash
# 檢查後端是否在運行
curl http://localhost:8000/api/health

# 應該返回 JSON（即使 status: degraded）
```

### 沒有可用的模型
- 檢查 `.env` 中是否有設定 API key
- 重新啟動後端
- 訪問 http://localhost:8000/api/models 確認

### 生成參數失敗
- 檢查 API key 是否有效
- 檢查是否超過 rate limit
- 查看後端終端機的錯誤訊息
- 試著換一個不同的模型

## Demo 技巧

### 預先準備
1. **測試不同提示詞**，找出效果好的範例
2. **準備幾個音訊檔案**（雖然 Phase 1 不會處理）
3. **確認 API key 有額度**
4. **預先打開瀏覽器**到 localhost:5173

### Demo 時
1. 先展示簡單的提示詞（如 "warm reverb"）
2. 再展示複雜的提示詞（如 "cathedral reverb with bright EQ and punchy compression"）
3. 展示切換不同 LLM 模型
4. 展示不同效果參數的差異
5. 展開 JSON 顯示給技術觀眾看

### 推薦的 Demo 提示詞

簡單範例：
- "warm and spacious"
- "bright and clear"
- "dark and moody"

進階範例：
- "after rain campus in October" (你的原始範例)
- "large cathedral with long reverb tail"
- "tight club mix with punchy compression"
- "vintage radio sound with midrange boost"
- "spacious concert hall with natural acoustics"

## 已知限制（Phase 1）

✅ **已實作：**
- 文字轉參數生成
- 多 LLM 支援
- 音訊檔案上傳
- 漂亮的 UI 展示

❌ **未實作（Phase 2）：**
- 音訊實際處理（需要 fx-processor 整合）
- 參數套用到音訊
- 音訊播放和比較
- Judge system 和迭代優化

## 下一步（未來延伸）

Phase 2 會加入：
1. 音訊實際處理（使用 fx-processor）
2. A/B 比較播放器
3. Judge system 評分
4. 參數迭代優化
5. 即時參數調整

## 緊急聯絡

如果明天 demo 有問題：
1. 檢查終端機的錯誤訊息
2. 訪問 http://localhost:8000/docs 測試 API
3. 檢查 `.env` 檔案中的 API keys
4. 重新啟動前後端

## 檔案結構參考

```
webapp/
├── backend/
│   ├── main.py                    # FastAPI 主程式
│   ├── generation_wrapper.py      # 簡化版參數生成
│   ├── requirements.txt
│   └── uploads/                   # 上傳的音訊檔案
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # 主要 React 組件
│   │   └── App.css               # 樣式
│   └── package.json
├── README.md                      # 完整文檔
├── QUICKSTART.md                  # 本檔案
└── start.sh                       # 啟動腳本
```

祝 demo 順利！ 🚀
