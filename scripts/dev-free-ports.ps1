# Free Vino Svoe local frontend/backend ports and leftover vite/uvicorn trees.
$ports = @(8091, 8092)

Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object {
    $_.CommandLine -and (
      $_.CommandLine -match 'uvicorn app\.main:app' -or
      $_.CommandLine -match 'multiprocessing\.spawn' -or
      $_.CommandLine -match 'vite' -or
      $_.CommandLine -match 'npm run dev'
    ) -and (
      $_.CommandLine -match 'Vino2026|vino-svoe' -or
      $_.CommandLine -match 'port 8091|port 8092|--port 8091|--port 8092'
    )
  } |
  ForEach-Object {
    Write-Host "Stopping PID $($_.ProcessId) (dev server tree)"
    & taskkill.exe /F /PID $_.ProcessId /T 2>$null | Out-Null
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }

foreach ($port in $ports) {
  $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
  foreach ($conn in $listeners) {
    $procId = $conn.OwningProcess
    if (-not $procId) { continue }
    Write-Host "Stopping PID $procId on port $port"
    & taskkill.exe /F /PID $procId /T 2>$null | Out-Null
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
  }
}

Start-Sleep -Seconds 1
foreach ($port in $ports) {
  $still = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
  if ($still.Count -gt 0) {
    Write-Host "WARNING: port $port still in use by PID(s): $($still.OwningProcess -join ', ')"
  } else {
    Write-Host "port $port free"
  }
}
