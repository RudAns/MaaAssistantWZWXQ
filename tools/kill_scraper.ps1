$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'update_card_db|dbg_menu' }
foreach ($p in $procs) {
    Stop-Process -Id $p.ProcessId -Force
    Write-Output ("killed " + $p.ProcessId)
}
if (-not $procs) { Write-Output "no matching python process" }
