# LaTeX 草稿（可选）

这里保留原有 XeLaTeX + `ctexart` 论文骨架。若团队最终使用 Word，可以只把它作为章节结构参考；若使用 LaTeX，可执行：

```powershell
cd paper\drafts\latex
latexmk -xelatex main.tex
```

该模板不替代组委会或赛区发布的官方页面。电子论文不要加入承诺书和编号专用页。

Q1–Q4 各自使用独立章节文件，便于模块 owner 并行写作；论文整合 owner 统一维护 `main.tex`、摘要和结论，避免频繁冲突。
