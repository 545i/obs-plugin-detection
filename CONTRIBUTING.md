# 貢獻指南 / Contributing

## ⚠️ 請勿直接在 `main` 編輯或推送

`main` 是受保護的發布分支。**所有變更都必須透過 Pull Request 進來。**

### 外部貢獻者（沒有寫入權限）

1. **Fork** 本倉庫到你自己的帳號。
2. 在你的 fork 開一個分支：`git switch -c feat/你的功能`。
3. 修改、提交、推到你的 fork。
4. 回到本倉庫發 **Pull Request**（目標 `main`）。

### 有協作權限者

不要直接 push `main`。一律開分支 → 發 PR → 審核通過後合併。

### 分支命名

- `feat/...` 新功能
- `fix/...` 修錯
- `docs/...` 文件

---

## English

`main` is protected — **all changes must go through a Pull Request**. External
contributors: **fork** the repo, branch on your fork, and open a PR against
`main`. Collaborators: never push to `main` directly; open a feature branch and a
PR. Branch prefixes: `feat/`, `fix/`, `docs/`.
