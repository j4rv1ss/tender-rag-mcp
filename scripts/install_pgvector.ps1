# Build pgvector (PG18-compatible) with MSVC and install it into PostgreSQL 18.
# Needs: Visual Studio 2022 (VC++ tools), git, and admin for the final copy.
# This documents exactly what was run to set up this machine.

$ErrorActionPreference = "Stop"
$work   = "D:\pgvector_build"
$pgRoot = "C:\Program Files\PostgreSQL\18"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

New-Item -ItemType Directory -Force $work | Out-Null
Set-Location $work
if (-not (Test-Path "$work\pgvector\.git")) {
  # master/HEAD carries the PG18 guards (vacuum_delay_point(bool)); v0.8.0 does not.
  git clone --depth 1 https://github.com/pgvector/pgvector.git
}

# Compile vector.dll against the PG18 server headers.
$build = "call `"$vcvars`" && set `"PGROOT=$pgRoot`" && cd /d `"$work\pgvector`" && nmake /F Makefile.win"
cmd /c $build
if (-not (Test-Path "$work\pgvector\vector.dll")) { throw "build failed: vector.dll missing" }

# Install (needs admin): copy the DLL + control + SQL into PG18.
$copy = @"
Copy-Item '$work\pgvector\vector.dll' '$pgRoot\lib\' -Force
Copy-Item '$work\pgvector\vector.control' '$pgRoot\share\extension\' -Force
Copy-Item '$work\pgvector\sql\vector--*.sql' '$pgRoot\share\extension\' -Force
"@
$tmp = Join-Path $work "do_install.ps1"
$copy | Out-File $tmp -Encoding utf8
Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$tmp

if (Test-Path "$pgRoot\lib\vector.dll") { Write-Output "pgvector installed. Now: CREATE EXTENSION vector;" }
else { throw "install copy failed (admin denied?)" }
