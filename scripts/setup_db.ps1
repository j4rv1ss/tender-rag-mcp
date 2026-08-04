# Create the tender_rag database and apply db/init.sql (extension + schema).
# Reads the Postgres password from tender_rag\.env (POSTGRES_PASSWORD).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup_db.ps1

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$psql  = "C:\Program Files\PostgreSQL\18\bin\psql.exe"
$envf  = Join-Path $root ".env"

if (-not (Test-Path $envf)) { throw "no .env at $envf - copy .env.example and set POSTGRES_PASSWORD" }
$cfg = @{}
Get-Content $envf | ForEach-Object {
  if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { $cfg[$matches[1]] = $matches[2].Trim() }
}
$db   = if ($cfg.POSTGRES_DB)   { $cfg.POSTGRES_DB }   else { "tender_rag" }
$user = if ($cfg.POSTGRES_USER) { $cfg.POSTGRES_USER } else { "postgres" }
$pass = $cfg.POSTGRES_PASSWORD
$pgHost = if ($cfg.POSTGRES_HOST) { $cfg.POSTGRES_HOST } else { "localhost" }
$port = if ($cfg.POSTGRES_PORT) { $cfg.POSTGRES_PORT } else { "5432" }
if (-not $pass -or $pass -eq "CHANGE_ME") { throw "set POSTGRES_PASSWORD in .env" }
$env:PGPASSWORD = $pass

Write-Output "creating database '$db' (if absent)..."
$exists = & $psql -U $user -h $pgHost -p $port -d postgres -tAc `
  "SELECT 1 FROM pg_database WHERE datname='$db'"
if ($exists -ne "1") {
  & $psql -U $user -h $pgHost -p $port -d postgres -c "CREATE DATABASE $db" | Out-Null
  Write-Output "  created."
} else { Write-Output "  already exists." }

Write-Output "applying db/init.sql (extension + tables + indexes)..."
& $psql -U $user -h $pgHost -p $port -d $db -v ON_ERROR_STOP=1 -f (Join-Path $root "db\init.sql")

Write-Output "verifying..."
& $psql -U $user -h $pgHost -p $port -d $db -tAc `
  "SELECT extname FROM pg_extension WHERE extname='vector'"
& $psql -U $user -h $pgHost -p $port -d $db -tAc `
  "SELECT 'tables: ' || string_agg(tablename,', ') FROM pg_tables WHERE schemaname='public'"
Write-Output "done."
