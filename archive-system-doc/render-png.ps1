<#
  render-png.ps1 —— 生成《档案库管理系统 业务流程框图》：矢量 SVG + 高清 PNG。

  思路：排版在 Node 里做（见 draw-archive-flow.mjs），但文字宽度用真实浏览器度量，
        所以中文换行、盒子高度都精确；最后由 Edge 无头模式把 SVG 光栅化成 2x PNG。

  用法： pwsh -File render-png.ps1
#>
$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path $edge)) { $edge = "C:\Program Files\Microsoft\Edge\Application\msedge.exe" }
if (-not (Test-Path $edge)) { throw "未找到 Microsoft Edge（SVG 仍可用，PNG 需浏览器光栅化）" }

$genPath = Join-Path $dir "draw-archive-flow.mjs"
$svgPath = Join-Path $dir "archive-flow.svg"
$pngPath = Join-Path $dir "archive-flow.png"
$utf8 = New-Object System.Text.UTF8Encoding($false)

# ---------- 1. Node 侧：抽取全部文案与字符集，写成 JSON 供页面注入 ----------
$htmlFile = Join-Path $dir "_build-html.tmp.html"
$sizeFile = Join-Path $dir "svg-size.json"
$stringsFile = Join-Path $dir "_strings.json"
$raw = & node $genPath
if ($LASTEXITCODE -ne 0) { throw "Node 排版脚本执行失败" }
Write-Host "[1/4] $raw"

$payload = [System.IO.File]::ReadAllText($stringsFile) | ConvertFrom-Json
Write-Host "      文案 $($payload.texts.Count) 条 / 待度量字符 $($payload.chars.Length) 个"

# ---------- 2. 浏览器：先量字，再让 Node 排版，最后把 SVG 回灌到 DOM ----------
$page = @"
<!doctype html><html><head><meta charset="utf-8"><title>build</title>
<style>html,body{margin:0;padding:0}#stage{position:absolute;left:-20000px;top:0}</style></head>
<body>
<pre id="log"></pre>
<div id="stage"></div>
<script>
  var MEA = null;
  (function () {
    var NS = "http://www.w3.org/2000/svg";
    var root = document.createElementNS(NS, "svg");
    root.setAttribute("width", "1200"); root.setAttribute("height", "300");
    root.setAttribute("style", "position:absolute;left:-20000px;top:0");
    var el = document.createElementNS(NS, "text");
    el.setAttribute("x", "10"); el.setAttribute("y", "150");
    root.appendChild(el); document.body.appendChild(root);
    var cache = {};
    MEA = function (s, size, weight, family, spacing) {
      var key = size + "|" + weight + "|" + family + "|" + spacing + "|" + s;
      if (key in cache) return cache[key];
      el.setAttribute("font-size", size);
      el.setAttribute("font-weight", weight);
      el.setAttribute("font-family", family);
      el.setAttribute("letter-spacing", spacing || 0);
      el.textContent = s;
      var w = el.getComputedTextLength();
      cache[key] = w;
      return w;
    };
  })();
  window.__MEA__ = function (s, size, weight, family, spacing) { return MEA(s, size, weight, family, spacing); };
  // __INJECT__
</script>
<script type="module">
  globalThis.__MEA__ = window.__MEA__;
  var log = document.getElementById("log");
  try {
    await import("./_svgdom.mjs");
    await import("./draw-archive-flow.mjs");
    var svg = await (await fetch("./archive-flow.svg?v=" + Date.now())).text();
    var stage = document.getElementById("stage");
    stage.innerHTML = svg;
    var node = stage.querySelector("svg");
    node.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    node.setAttribute("style", "display:block");
    log.textContent = "BUILD-OK " + node.getAttribute("viewBox");
    document.title = "BUILD-OK";
  } catch (e) {
    log.textContent = "BUILD-FAIL " + (e && (e.stack || e.message));
    document.title = "BUILD-FAIL";
  }
</script>
</body></html>
"@
[System.IO.File]::WriteAllText($htmlFile, $page, $utf8)

# 注入文案与字符集（JSON 化后直接替换占位符，避免任何转义问题）
$inject = "window.__TEXTS__ = " + (ConvertTo-Json $payload.texts -Compress) + ";" + [char]10 +
          "window.__CHARS__ = " + (ConvertTo-Json $payload.chars -Compress -AsArray) + ";" + [char]10
$page = [System.IO.File]::ReadAllText($htmlFile)
if (-not $page.Contains("// __INJECT__")) { throw "页面缺少注入占位符" }
$page = $page.Replace("// __INJECT__", $inject)
[System.IO.File]::WriteAllText($htmlFile, $page, $utf8)

$dump = & $edge --headless=new --disable-gpu --no-sandbox --virtual-time-budget=20000 --dump-dom "file:///$($htmlFile -replace '\\','/')" 2>&1 | Out-String
if ($dump -notmatch "BUILD-OK") {
  $m = [regex]::Match($dump, "BUILD-FAIL[^<]*")
  throw "页面构建失败：" + $m.Value
}
Write-Host "[2/4] SVG 已生成（浏览器实测宽度 + Node 排版）"

# ---------- 3. 用 CDP 精确渲染 PNG（绕开 --screenshot 的尺寸上限与打印字体回退） ----------
$size = [System.IO.File]::ReadAllText($sizeFile) | ConvertFrom-Json
$w = [int]$size.width; $h = [int]$size.height
$png2x = Join-Path $dir "archive-flow@2x.png"
$log = & node (Join-Path $dir "capture-png.mjs") $edge $svgPath $png2x $w $h 2
if ($LASTEXITCODE -ne 0) { throw "CDP 截图失败" }
Write-Host "[3/4] $log"

# 1x 版本：便于直接插入 Word / 网页
Add-Type -AssemblyName System.Drawing
$src = [System.Drawing.Image]::FromFile($png2x)
$bmp = New-Object System.Drawing.Bitmap $w, $h
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
$g.DrawImage($src, 0, 0, $w, $h)
$g.Dispose()
$bmp.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose(); $src.Dispose()

# 校验：标题文字必须真的画出来（曾遇到打印管线把粗体中文丢成空白）
$check = New-Object System.Drawing.Bitmap (New-Object System.Drawing.Bitmap $png2x)
$dark = 0
for ($y = 150; $y -lt 360; $y += 4) { for ($x = 2400; $x -lt 4100; $x += 4) { $c = $check.GetPixel($x, $y); if ($c.R -lt 140 -and $c.B -lt 160) { $dark++ } } }
$check.Dispose()
if ($dark -lt 500) { Write-Warning "标题区域深色像素仅 $dark，可能未正常渲染，请人工核对 archive-flow@2x.png" }
else { Write-Host "      标题区域校验通过（深色像素 $dark）" }

# ---------- 4. 收尾 ----------
Remove-Item (Join-Path $dir "_strings.json"), (Join-Path $dir "_build-html.tmp.html"), (Join-Path $dir "_render.tmp.html") -ErrorAction SilentlyContinue
Write-Host "[4/4] 完成：archive-flow.svg（矢量可编辑） / archive-flow.png ${w}x${h} / archive-flow@2x.png $((2*$w))x$((2*$h))"