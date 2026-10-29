# Progress at a glance.   powershell -ExecutionPolicy Bypass -File scripts\status.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
$done = Get-ChildItem runs -Recurse -Filter summary.json -ErrorAction SilentlyContinue
"finished runs: " + $done.Count + "   (targets seed 0: E3 44 | E1 80 | E2 17 | E7 2)"
$done | Group-Object { $_.Directory.Parent.Name } | ForEach-Object { "  " + $_.Name + ": " + $_.Count }
$r = Get-ChildItem runs -Recurse -Filter rounds.jsonl -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($r) {
  $n = (Get-Content $r.FullName).Count
  "current run: " + $r.Directory.Parent.Name + "\" + $r.Directory.Name + "   rounds: $n / 300   last write: " + $r.LastWriteTime
  if ($n -gt 0) { $last = Get-Content $r.FullName -Tail 1 | ConvertFrom-Json; "  last round $($last.round): time_s=$([math]::Round($last.time_s,1)) attack=$($last.attack_active)" }
}
$procs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 5 }
"training processes (should be 1-2): " + @($procs).Count
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used --format=csv,noheader
