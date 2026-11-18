<#
.SYNOPSIS
    Pipelines Spark raw -> curated -> analytics en un contenedor de Azure Container Instances.

.DESCRIPTION
    a) Copia limpia del proyecto (robocopy) en una carpeta temporal y comprobación de tamaño.
    b) az acr build de la imagen de Spark (Dockerfile.spark).
    c) Obtiene la cadena de conexión del almacenamiento y la contraseña del registro
       (solo en memoria: no se escriben en disco ni se muestran).
    d) Lanza un contenedor de 2 vCPU y 8 GB que lee raw/ del lake y escribe curated/ y analytics/.
    e) Espera a que termine y muestra su estado final y su log.
    f) Sube docs/figures a analytics/figures.
    g) Lista los ficheros de curated/ y analytics/ en el lake con su tamaño.
    h) Borra el contenedor (también si algo falla a mitad, en finally).
    i) Descarga analytics_classifier_features y la compara con la tabla generada por
       Databricks (data/analytics/classifier_features.parquet): mismas filas y mismos
       valores por (match_key, player_index, n) con tolerancia 1e-9.

    Se detiene con un mensaje claro y código de salida 1 si falla cualquier paso
    (también si las tablas no coinciden).

.PARAMETER DryRun
    No llama a Azure: hace la copia local de verdad y muestra los comandos az que
    ejecutaría (con los secretos enmascarados). La comparación final se ejecuta de
    verdad contra la propia tabla de referencia. Sirve para probar el script en local.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_cloud_pipelines.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_cloud_pipelines.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$ResourceGroup = "rg-tfm-axelrod",
    [string]$Location = "swedencentral",
    [string]$StorageAccount = "sttfmaxelrodpbp",
    [string]$FileSystem = "lake",
    [string]$Registry = "acrtfmaxelrodpbp",
    [string]$Image = "axelrod-spark:v1",
    [string]$ContainerName = "aci-axelrod-spark",
    [int]$Cpu = 2,
    [int]$MemoryGB = 8,
    [string]$DriverMemory = "5g",
    [string]$ReferenceTable = "data\analytics\classifier_features.parquet",
    [string]$FiguresDir = "docs\figures",
    [double]$Tolerance = 1e-9,
    [int]$TimeoutMinutes = 90,
    [int]$PollSeconds = 30,
    [int]$MaxContextMB = 20,
    [switch]$SkipBuild,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Secrets = New-Object System.Collections.Generic.List[string]
$Created = New-Object System.Collections.Generic.List[string]
$BuildDir = $null
$StageDir = $null
$Start = Get-Date
$Tables = [ordered]@{
    "curated_moves"                 = "curated"
    "curated_evolution"             = "curated"
    "analytics_match_summary"       = "analytics"
    "analytics_strategy_ranking"    = "analytics"
    "analytics_classifier_features" = "analytics"
}
$FeaturesPrefix = "analytics/analytics_classifier_features"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Hide-Secrets([string]$Text) {
    foreach ($s in $Secrets) { if ($s) { $Text = $Text.Replace($s, "********") } }
    return $Text
}

function Invoke-Az {
    <# Ejecuta az, comprueba el código de salida y devuelve la salida (o la muestra con -Stream). #>
    param(
        [Parameter(Mandatory = $true)][string]$Step,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$Stream,
        $DryRunResult = ""
    )
    if ($DryRun) {
        Write-Host ("  [dry-run] az " + (Hide-Secrets ($Arguments -join " "))) -ForegroundColor DarkGray
        return $DryRunResult
    }
    if ($Stream) {
        & az @Arguments | Out-Host
    } else {
        $out = & az @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Ha fallado el paso '$Step' (az $($Arguments[0]) $($Arguments[1]), código $LASTEXITCODE)."
    }
    if (-not $Stream) { return $out }
}

function Assert-SafeForCmd([string]$Value, [string]$What) {
    # az en Windows es un .cmd: estos caracteres romperían el paso de argumentos.
    if ([string]::IsNullOrWhiteSpace($Value)) { throw "No se ha podido obtener $What (vacío)." }
    if ($Value -match '[\s"&|<>^%!]') {
        throw "$What contiene caracteres que az.cmd no puede recibir de forma segura como argumento."
    }
}

function Get-LakeFiles([string]$Prefix, [string]$DryRunJson) {
    <# Ficheros (no carpetas) bajo $Prefix con su tamaño, como objetos {name, size}. #>
    $listing = Invoke-Az -Step "listar $FileSystem/$Prefix" -DryRunResult $DryRunJson -Arguments @(
        "storage", "fs", "file", "list", "--file-system", $FileSystem, "--path", $Prefix,
        "--recursive", "true", "--connection-string", $conn,
        "--query", "[?!isDirectory].{name:name,size:contentLength}", "-o", "json", "--only-show-errors")
    # En PowerShell 5.1 ConvertFrom-Json devuelve el array como un único objeto: se desenrolla.
    $parsed = ($listing -join "`n") | ConvertFrom-Json
    return @($parsed | ForEach-Object { $_ } | Where-Object { $null -ne $_ })
}

function Show-LakeFiles([string]$Prefix, $Items) {
    $totalBytes = 0.0
    foreach ($it in $Items) { $totalBytes += [double]$it.size }
    if ($Items.Count -gt 0) {
        $Items | Sort-Object name | Format-Table @{ n = "Fichero"; e = { $_.name } },
            @{ n = "Tamaño (MB)"; e = { "{0:N2}" -f ([double]$_.size / 1MB) }; a = "Right" } -AutoSize | Out-Host
    }
    Write-Host ("  {0}/: {1} ficheros, {2:N2} MB en total" -f $Prefix, $Items.Count, ($totalBytes / 1MB))
}

function Get-FakeListing([string]$Prefix) {
    # Listado simulado para -DryRun: un part-*.parquet por tabla de la capa y las figuras.
    $rows = @()
    foreach ($t in $Tables.Keys) {
        if ($Tables[$t] -eq $Prefix) { $rows += '{{"name":"{0}/{1}/part-00000-dryrun.snappy.parquet","size":{2}}}' -f $Prefix, $t, 1048576 }
    }
    if ($Prefix -eq "analytics") {
        foreach ($f in $figures) { $rows += '{{"name":"analytics/figures/{0}","size":{1}}}' -f $f.Name, $f.Length }
    }
    return "[" + ($rows -join ",") + "]"
}

function Remove-Containers {
    foreach ($name in @($Created)) {
        try {
            Invoke-Az -Step "borrar $name" -Arguments @(
                "container", "delete", "-g", $ResourceGroup, "-n", $name, "--yes", "-o", "none", "--only-show-errors") | Out-Null
            Write-Host "  borrado $name"
            $Created.Remove($name) | Out-Null
        } catch {
            Write-Host "  No se pudo borrar ${name}: bórralo a mano con 'az container delete -g $ResourceGroup -n $name --yes'" -ForegroundColor Yellow
        }
    }
}

$failed = $false
try {
    # ------------------------------------------------------------------ #
    Write-Step "0) Comprobaciones previas"
    if (-not $DryRun -and -not (Get-Command az -ErrorAction SilentlyContinue)) {
        throw "No se encuentra la CLI de Azure (az) en el PATH."
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "No se encuentra uv en el PATH (hace falta para la comparación final)."
    }
    $refPath = Join-Path $ProjectRoot $ReferenceTable
    if (-not (Test-Path $refPath)) { throw "No existe la tabla de referencia de Databricks: $refPath" }
    $figPath = Join-Path $ProjectRoot $FiguresDir
    $figures = @(Get-ChildItem -Path $figPath -File -ErrorAction SilentlyContinue)
    if ($figures.Count -eq 0) { throw "No hay figuras en $figPath" }
    foreach ($f in $figures) { Assert-SafeForCmd $f.Name "El nombre de la figura '$($f.Name)'" }
    if ($ContainerName -notmatch '^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$') { throw "Nombre de contenedor no válido: $ContainerName" }
    if ($DriverMemory -notmatch '^\d+[gm]$') { throw "DriverMemory no válida: $DriverMemory (p. ej. 5g)" }
    Write-Host "  Referencia: $ReferenceTable ($('{0:N1}' -f ((Get-Item $refPath).Length / 1MB)) MB); figuras: $($figures.Count)"

    # Carpeta de trabajo sin espacios (az.cmd recibe estas rutas como argumentos).
    $StageDir = Join-Path $env:TEMP ("tfm-axelrod-spark-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    Assert-SafeForCmd $StageDir "La carpeta temporal ($StageDir)"
    New-Item -ItemType Directory -Path $StageDir | Out-Null

    $sub = Invoke-Az -Step "comprobar sesión de az" -DryRunResult "(dry-run)" `
        -Arguments @("account", "show", "--query", "name", "-o", "tsv", "--only-show-errors")
    Write-Host "  Suscripción: $sub"
    Invoke-Az -Step "comprobar grupo de recursos" -DryRunResult "" `
        -Arguments @("group", "show", "-n", $ResourceGroup, "-o", "none", "--only-show-errors") | Out-Null
    Write-Host "  Grupo de recursos: $ResourceGroup ($Location)"

    # ------------------------------------------------------------------ #
    if ($SkipBuild) {
        Write-Step "a-b) Se omite la construcción de la imagen (-SkipBuild)"
    } else {
        Write-Step "a) Copia limpia del proyecto para el contexto de construcción"
        $BuildDir = Join-Path $env:TEMP ("tfm-axelrod-build-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        $excludeDirs = @(".git", ".venv", "data", "mlruns", "docs", "notebooks", "pipelines",
                         "tests", "models", "__pycache__", ".pytest_cache")
        & robocopy $ProjectRoot $BuildDir /E /XD @excludeDirs /XF mlflow.db /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy ha fallado (código $LASTEXITCODE)." }
        foreach ($required in @("Dockerfile.spark", "requirements-spark.txt", "tfm_axelrod\spark_pipelines.py", "tfm_axelrod\cloud.py")) {
            if (-not (Test-Path (Join-Path $BuildDir $required))) { throw "Falta $required en la copia." }
        }
        $files = @(Get-ChildItem -Path $BuildDir -Recurse -File)
        $sizeMB = ($files | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host ("  Copia en {0}: {1} ficheros, {2:N2} MB" -f $BuildDir, $files.Count, $sizeMB)
        if ($sizeMB -gt $MaxContextMB) {
            throw ("La copia pesa {0:N1} MB (> {1} MB): revisa las exclusiones antes de subirla." -f $sizeMB, $MaxContextMB)
        }

        Write-Step "b) az acr build -> $Registry/$Image"
        Invoke-Az -Step "construir la imagen" -Stream -Arguments @(
            "acr", "build", "--registry", $Registry, "--image", $Image,
            "--file", "Dockerfile.spark", $BuildDir)
    }

    # ------------------------------------------------------------------ #
    Write-Step "c) Credenciales (solo en memoria)"
    $loginServer = Invoke-Az -Step "obtener el servidor del registro" -DryRunResult "$Registry.azurecr.io" `
        -Arguments @("acr", "show", "-n", $Registry, "--query", "loginServer", "-o", "tsv", "--only-show-errors")
    $acrUser = Invoke-Az -Step "obtener el usuario del registro" -DryRunResult $Registry `
        -Arguments @("acr", "credential", "show", "-n", $Registry, "--query", "username", "-o", "tsv", "--only-show-errors")
    $acrPassword = Invoke-Az -Step "obtener la contraseña del registro" -DryRunResult "fake-acr-password" `
        -Arguments @("acr", "credential", "show", "-n", $Registry, "--query", "passwords[0].value", "-o", "tsv", "--only-show-errors")
    $conn = Invoke-Az -Step "obtener la cadena de conexión" `
        -DryRunResult "DefaultEndpointsProtocol=https;AccountName=$StorageAccount;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net" `
        -Arguments @("storage", "account", "show-connection-string", "-g", $ResourceGroup, "-n", $StorageAccount,
                     "--query", "connectionString", "-o", "tsv", "--only-show-errors")
    $loginServer = "$loginServer".Trim(); $acrUser = "$acrUser".Trim()
    $acrPassword = "$acrPassword".Trim(); $conn = "$conn".Trim()
    $Secrets.Add($acrPassword); $Secrets.Add($conn)
    Assert-SafeForCmd $acrPassword "La contraseña del registro"
    Assert-SafeForCmd $conn "La cadena de conexión"
    Write-Host "  Registro: $loginServer (usuario $acrUser). Secretos obtenidos, no se muestran."

    # ------------------------------------------------------------------ #
    Write-Step "d) Lanzando $ContainerName ($Cpu vCPU, $MemoryGB GB)"
    $cmd = "python -m tfm_axelrod.spark_pipelines --from-lake --filesystem $FileSystem"
    Invoke-Az -Step "crear el contenedor $ContainerName" -Arguments @(
        "container", "create", "-g", $ResourceGroup, "-n", $ContainerName, "--location", $Location,
        "--image", "$loginServer/$Image", "--os-type", "Linux", "--cpu", "$Cpu", "--memory", "$MemoryGB",
        "--restart-policy", "Never",
        "--registry-login-server", $loginServer, "--registry-username", $acrUser,
        "--registry-password", $acrPassword,
        "--environment-variables", "SPARK_DRIVER_MEMORY=$DriverMemory",
        "--secure-environment-variables", "AZURE_STORAGE_CONNECTION_STRING=$conn",
        "--command-line", $cmd, "--no-wait", "-o", "none", "--only-show-errors") | Out-Null
    $Created.Add($ContainerName)
    Write-Host "  $ContainerName -> $cmd"

    # ------------------------------------------------------------------ #
    Write-Step "e) Esperando a que termine (máximo $TimeoutMinutes min, consulta cada $PollSeconds s)"
    $query = "{group:instanceView.state,prov:provisioningState,container:containers[0].instanceView.currentState.state," +
             "exit:containers[0].instanceView.currentState.exitCode,detail:containers[0].instanceView.currentState.detailStatus}"
    $final = $null
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ($true) {
        $raw = Invoke-Az -Step "consultar el estado de $ContainerName" `
            -DryRunResult '{"group":"Succeeded","prov":"Succeeded","container":"Terminated","exit":0,"detail":"Completed"}' `
            -Arguments @("container", "show", "-g", $ResourceGroup, "-n", $ContainerName, "--query", $query, "-o", "json", "--only-show-errors")
        $st = ($raw -join "`n") | ConvertFrom-Json
        $label = "$($st.prov)/$($st.group)/$($st.container)"
        Write-Host ("  [{0:hh\:mm\:ss}] {1}: {2}" -f ((Get-Date) - $Start), $ContainerName, $label)
        $done = ($st.container -eq "Terminated") -or ($st.prov -eq "Failed") -or
                (@("Succeeded", "Failed", "Stopped") -contains $st.group)
        if ($done) {
            $final = [pscustomobject]@{ ok = (($st.prov -ne "Failed") -and ($st.exit -eq 0)); exit = $st.exit; detail = $st.detail }
            break
        }
        if ((Get-Date) -gt $deadline) { break }
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Step "Estado final y log"
    if ($null -ne $final) {
        $color = if ($final.ok) { "Green" } else { "Red" }
        Write-Host ("  {0}: {1} (código de salida {2}, {3})" -f $ContainerName, $(if ($final.ok) { "OK" } else { "FALLO" }), $final.exit, $final.detail) -ForegroundColor $color
    } else {
        Write-Host "  ${ContainerName}: sin terminar al agotarse el tiempo" -ForegroundColor Red
    }
    Write-Host "  ----- log de $ContainerName -----" -ForegroundColor DarkCyan
    Invoke-Az -Step "leer el log de $ContainerName" -Stream -DryRunResult "" `
        -Arguments @("container", "logs", "-g", $ResourceGroup, "-n", $ContainerName, "--only-show-errors")
    if ($null -eq $final -or -not $final.ok) {
        throw "El contenedor $ContainerName ha fallado o no ha terminado. Revisa su log arriba."
    }

    # ------------------------------------------------------------------ #
    Write-Step "f) Subiendo $($figures.Count) figuras a $FileSystem/analytics/figures"
    $figStage = Join-Path $StageDir "figures"
    New-Item -ItemType Directory -Path $figStage | Out-Null
    foreach ($f in $figures) {
        $src = Join-Path $figStage $f.Name
        Copy-Item -LiteralPath $f.FullName -Destination $src
        Invoke-Az -Step "subir la figura $($f.Name)" -Arguments @(
            "storage", "fs", "file", "upload", "--file-system", $FileSystem,
            "--path", "analytics/figures/$($f.Name)", "--source", $src, "--overwrite", "true",
            "--connection-string", $conn, "-o", "none", "--only-show-errors") | Out-Null
        Write-Host ("  subida analytics/figures/{0} ({1:N2} MB)" -f $f.Name, ($f.Length / 1MB))
    }

    # ------------------------------------------------------------------ #
    Write-Step "g) Ficheros en $FileSystem/curated y $FileSystem/analytics"
    $all = @()
    foreach ($layer in @("curated", "analytics")) {
        $items = Get-LakeFiles $layer (Get-FakeListing $layer)
        Show-LakeFiles $layer $items
        $all += $items
    }
    foreach ($t in $Tables.Keys) {
        $prefix = "$($Tables[$t])/$t/"
        $n = @($all | Where-Object { $_.name.StartsWith($prefix) -and $_.name.EndsWith(".parquet") }).Count
        if ($n -eq 0) { throw "La tabla $prefix no tiene ficheros Parquet en el lake." }
    }
    $nFig = @($all | Where-Object { $_.name.StartsWith("analytics/figures/") }).Count
    if ($nFig -lt $figures.Count) { throw "En analytics/figures hay $nFig ficheros y se esperaban $($figures.Count)." }

    # ------------------------------------------------------------------ #
    Write-Step "h) Borrando el contenedor"
    Remove-Containers

    # ------------------------------------------------------------------ #
    Write-Step "i) Comparando analytics_classifier_features con la tabla de Databricks"
    $dl = Join-Path $StageDir "analytics_classifier_features"
    New-Item -ItemType Directory -Path $dl | Out-Null
    $parts = @($all | Where-Object { $_.name.StartsWith("$FeaturesPrefix/") -and $_.name.EndsWith(".parquet") })
    $i = 0
    foreach ($p in $parts) {
        $i++
        $dest = Join-Path $dl ("part-{0:D5}.parquet" -f $i)
        Invoke-Az -Step "descargar $($p.name)" -Arguments @(
            "storage", "fs", "file", "download", "--file-system", $FileSystem, "--path", $p.name,
            "--destination", $dest, "--overwrite", "true", "--connection-string", $conn,
            "-o", "none", "--only-show-errors") | Out-Null
    }
    if ($DryRun) {
        # Sin Azure no hay nada que descargar: se compara la referencia consigo misma
        # para ejercitar de verdad el paso de comparación.
        Copy-Item -LiteralPath $refPath -Destination (Join-Path $dl "part-00001.parquet") -Force
        Write-Host "  [dry-run] se compara la tabla de referencia consigo misma" -ForegroundColor DarkGray
    }
    Write-Host "  Descargados $($parts.Count) ficheros en $dl"
    Push-Location $ProjectRoot
    try {
        & uv run --quiet python -m tfm_axelrod.compare_tables $dl $refPath --tol $Tolerance | Out-Host
        $cmpExit = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($cmpExit -eq 0) {
        Write-Host ""
        Write-Host "  COINCIDEN: la tabla generada en ACI es igual a la de Databricks (tolerancia $Tolerance)." -ForegroundColor Green
    } else {
        throw "NO COINCIDEN: la tabla generada en ACI difiere de la de Databricks (detalle arriba)."
    }
}
catch {
    $failed = $true
    Write-Host ""
    Write-Host ("ERROR: " + (Hide-Secrets $_.Exception.Message)) -ForegroundColor Red
}
finally {
    # ------------------------------------------------------------------ #
    if ($Created.Count -gt 0) {
        Write-Step "Borrando $($Created.Count) contenedor(es) pendiente(s)"
        Remove-Containers
    }
    foreach ($d in @($BuildDir, $StageDir)) { if ($d -and (Test-Path $d)) { Remove-Item -Recurse -Force $d } }
    Remove-Variable -Name conn, acrPassword -ErrorAction SilentlyContinue
    $Secrets.Clear()
    Write-Host ("`nDuración total: {0:hh\:mm\:ss}" -f ((Get-Date) - $Start))
}

if ($failed) { exit 1 }
Write-Host "Pipelines Spark en ACI completados y verificados." -ForegroundColor Green
exit 0
