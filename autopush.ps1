# Configuración: Hora de término (14:00 / 2:00 PM)
$HoraFin = Get-Date -Hour 14 -Minute 0 -Second 0
$MinutosEspera = 30

Write-Host "[INICIO] Iniciando script de Auto-Push." -ForegroundColor Green
Write-Host "[INFO] Se detendra automaticamente a las: $HoraFin" -ForegroundColor Cyan

# Bucle que se ejecuta mientras la hora actual sea menor a la hora fin
while ((Get-Date) -lt $HoraFin) {
    $HoraActual = Get-Date -Format "HH:mm:ss"
    
    Write-Host "[$HoraActual] >> Ejecutando Git Push..." -ForegroundColor Yellow

    # Comandos de Git
    git add .
    
    # Hacemos commit
    git commit -m "Auto-push: Train RAINBOW $HoraActual"
    
    git push

    Write-Host "[OK] Hecho. Esperando $MinutosEspera minutos para la siguiente ejecucion..." -ForegroundColor Gray
    
    # Pausa el script por 30 minutos (30 * 60 segundos)
    Start-Sleep -Seconds ($MinutosEspera * 60)
}

Write-Host "[FIN] Son las 14:00. El script ha terminado." -ForegroundColor Red