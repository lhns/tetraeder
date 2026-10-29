# Reset vendor/llama.cpp to the pin (or $env:PIN_REF) and apply patches/series in order. Idempotent.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$sub = Join-Path $root 'vendor/llama.cpp'
$ref = $env:PIN_REF
if (-not $ref) {
    $ref = (Get-Content (Join-Path $root 'vendor/PIN') | Where-Object { $_ -like 'commit=*' }) -replace '^commit=', ''
}
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { throw "git $args failed" } }

Invoke-Git -C $root submodule update --init --depth 1 vendor/llama.cpp
& git -C $sub cat-file -e "$ref^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-Git -C $sub fetch --depth 1 origin $ref
    $ref = 'FETCH_HEAD'
}
Invoke-Git -C $sub checkout -q --detach $ref
Invoke-Git -C $sub reset -q --hard
Invoke-Git -C $sub clean -qfdx

$n = 0
foreach ($line in Get-Content (Join-Path $root 'patches/series')) {
    $p = ($line -replace '#.*$', '').Trim()
    if (-not $p) { continue }
    Write-Host "apply $p"
    Invoke-Git -C $sub apply --index (Join-Path $root "patches/$p")
    $n++
}
Write-Host "ok: $(git -C $sub rev-parse --short HEAD) + $n patches"
