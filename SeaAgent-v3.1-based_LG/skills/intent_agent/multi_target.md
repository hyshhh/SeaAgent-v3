# IntentAgent · 多目标拆分

## 何时使用
问题中出现多艘船、多个舷号或多个外观描述，并用顿号、逗号、“和/与/及/以及”等并列。

## 规则
1. 必须输出 `targetItems` 数组，**每个目标一条**，禁止合并成一个 `description`/`hullNumber` 字符串。
2. 舷号形态（字母数字组合、明确“舷号”标注）→ `kind=hull`，填 `hullNumber` 与 `label`。
3. 外观/颜色/船型描述 → `kind=description`，填 `description` 与 `label`。
4. 可调用 `parseTargets` 作切分参考，最终以你确认的 `targetItems` 为准。
5. 单目标时 `targetItems` 可为空数组或仅一项；多目标时 `len(targetItems) > 1`。
6. 多目标时不要只填一个 `hullNumber` 而丢掉其余目标。

## 示例
- “003、0123、A01 是否出现” → 三个 hull 项
- “黄色无人艇和白色快艇” → 两个 description 项
