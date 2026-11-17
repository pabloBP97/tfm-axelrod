<#
.SYNOPSIS
    Simulación distribuida en Azure Container Instances: un contenedor por nivel de ruido.

.DESCRIPTION
    a) Copia limpia del proyecto (robocopy) en una carpeta temporal y comprobación de tamaño.
    b) az acr build de la imagen del simulador (Dockerfile.simulate).
    c) Obtiene la cadena de conexión del almacenamiento y la contraseña del registro
       (solo en memoria: no se escriben en disco ni se muestran).
    d) Lanza un contenedor por nivel de ruido, en paralelo (--no-wait).
    e) Espera a que terminen y muestra el estado final y el log de cada uno.
    f) Lista los ficheros subidos al lake bajo el prefijo indicado.
    g) Borra los grupos de contenedores (también si algo falla a mitad).

    Se detiene con un mensaje claro y código de salida 1 si falla cualquier paso.

.PARAMETER DryRun
    No llama a Azure: hace la copia local de verdad y muestra los comandos az que
    ejecutaría (con los secretos enmascarados). Sirve para probar el script en local.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_cloud_simulation.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\run_cloud_simulation.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$ResourceGroup = "rg-tfm-axelrod",
    [string]$Location = "swedencentral",
    [string]$StorageAccount = "sttfmaxelrodpbp",
    [string]$FileSystem = "lake",
    [string]$Registry = "acrtfmaxelrodpbp",
    [string]$Image = "axelrod-sim:v1",
    [string]$UploadPrefix = "raw_cloud",
    [string[]]$Noises = @("0", "0.01", "0.05", "0.1"),
    [string]$SimArgs = "--preset full --turns 100 200 --repetitions 10 --seed 42",
    [int]$TimeoutMinutes = 120,
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
$Start = Get-Date

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

function Get-ContainerName([string]$Noise) {
    return "aci-axelrod-sim-n" + ($Noise -replace '[^0-9]', '')
}

$failed = $false
try {
    # ------------------------------------------------------------------ #
    Write-Step "0) Comprobaciones previas"
    if (-not $DryRun -and -not (Get-Command az -ErrorAction SilentlyContinue)) {
        throw "No se encuentra la CLI de Azure (az) en el PATH."
    }
    # Con -File, "-Noises 0,0.1" llega como una sola cadena: se separa por comas.
    $Noises = @($Noises | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $inv = [Globalization.CultureInfo]::InvariantCulture
    foreach ($n in $Noises) {
        $v = 0.0
        if (-not [double]::TryParse($n, [Globalization.NumberStyles]::Float, $inv, [ref]$v) -or $v -lt 0 -or $v -gt 1) {
            throw "Nivel de ruido no válido: '$n' (debe ser un número entre 0 y 1, con punto decimal)."
        }
    }
    if ($Noises.Count -eq 0) { throw "No se ha indicado ningún nivel de ruido." }
    $names = @($Noises | ForEach-Object { Get-ContainerName $_ })
    if (@($names | Select-Object -Unique).Count -ne $names.Count) {
        throw "Los niveles de ruido generan nombres de contenedor repetidos: $($names -join ', ')"
    }
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
        foreach ($required in @("Dockerfile.simulate", "requirements-simulate.txt", "tfm_axelrod\simulate.py", "tfm_axelrod\cloud.py")) {
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
            "--file", "Dockerfile.simulate", $BuildDir)
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
    Write-Step "d) Lanzando $($Noises.Count) contenedores en paralelo"
    foreach ($noise in $Noises) {
        $name = Get-ContainerName $noise
        $cmd = "python -m tfm_axelrod.simulate $SimArgs --noise $noise --out /tmp/sim --upload-prefix $UploadPrefix --upload-filesystem $FileSystem"
        Invoke-Az -Step "crear el contenedor $name" -Arguments @(
            "container", "create", "-g", $ResourceGroup, "-n", $name, "--location", $Location,
            "--image", "$loginServer/$Image", "--os-type", "Linux", "--cpu", "1", "--memory", "2",
            "--restart-policy", "Never",
            "--registry-login-server", $loginServer, "--registry-username", $acrUser,
            "--registry-password", $acrPassword,
            "--secure-environment-variables", "AZURE_STORAGE_CONNECTION_STRING=$conn",
            "--command-line", $cmd, "--no-wait", "-o", "none", "--only-show-errors") | Out-Null
        $Created.Add($name)
        Write-Host "  $name -> ruido $noise"
    }

    # ------------------------------------------------------------------ #
    Write-Step "e) Esperando a que terminen (máximo $TimeoutMinutes min, consulta cada $PollSeconds s)"
    $query = "{group:instanceView.state,prov:provisioningState,container:containers[0].instanceView.currentState.state," +
             "exit:containers[0].instanceView.currentState.exitCode,detail:containers[0].instanceView.currentState.detailStatus}"
    $final = @{}
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ($final.Count -lt $Created.Count) {
        $line = @()
        foreach ($name in $Created) {
            if ($final.ContainsKey($name)) { $line += "${name}: $($final[$name].label)"; continue }
            $raw = Invoke-Az -Step "consultar el estado de $name" `
                -DryRunResult '{"group":"Succeeded","prov":"Succeeded","container":"Terminated","exit":0,"detail":"Completed"}' `
                -Arguments @("container", "show", "-g", $ResourceGroup, "-n", $name, "--query", $query, "-o", "json", "--only-show-errors")
            $st = ($raw -join "`n") | ConvertFrom-Json
            $done = ($st.container -eq "Terminated") -or ($st.prov -eq "Failed") -or
                    (@("Succeeded", "Failed", "Stopped") -contains $st.group)
            $label = "$($st.prov)/$($st.group)/$($st.container)"
            if ($done) {
                $ok = ($st.prov -ne "Failed") -and ($st.exit -eq 0)
                $final[$name] = [pscustomobject]@{ ok = $ok; exit = $st.exit; label = $label; detail = $st.detail }
            }
            $line += "${name}: $label"
        }
        $elapsed = (Get-Date) - $Start
        Write-Host ("  [{0:hh\:mm\:ss}] {1}" -f $elapsed, ($line -join " | "))
        if ($final.Count -ge $Created.Count) { break }
        if ((Get-Date) -gt $deadline) { break }
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Step "Estado final y logs"
    foreach ($name in $Created) {
        if ($final.ContainsKey($name)) {
            $f = $final[$name]
            $color = if ($f.ok) { "Green" } else { "Red" }
            Write-Host ("  {0}: {1} (código de salida {2}, {3})" -f $name, $(if ($f.ok) { "OK" } else { "FALLO" }), $f.exit, $f.detail) -ForegroundColor $color
        } else {
            Write-Host "  ${name}: sin terminar al agotarse el tiempo" -ForegroundColor Red
        }
        Write-Host "  ----- log de $name -----" -ForegroundColor DarkCyan
        Invoke-Az -Step "leer el log de $name" -Stream -DryRunResult "" `
            -Arguments @("container", "logs", "-g", $ResourceGroup, "-n", $name, "--only-show-errors")
    }
    $bad = @($Created | Where-Object { -not $final.ContainsKey($_) -or -not $final[$_].ok })
    if ($bad.Count -gt 0) {
        throw "Han fallado o no han terminado $($bad.Count) contenedores: $($bad -join ', '). Revisa sus logs arriba."
    }

    # ------------------------------------------------------------------ #
    Write-Step "f) Ficheros en $FileSystem/$UploadPrefix"
    $fakeListing = "[" + (($Noises | ForEach-Object { $n = $_; 100, 200 | ForEach-Object {
        '{{"name":"{0}/roundrobin/noise={1}/turns={2}/ro_dryrun.parquet","size":{3}}}' -f $UploadPrefix, $n, $_, (1MB * $_ / 50) } }) -join ",") + "]"
    $listing = Invoke-Az -Step "listar el lake" -DryRunResult $fakeListing -Arguments @(
        "storage", "fs", "file", "list", "--file-system", $FileSystem, "--path", $UploadPrefix,
        "--recursive", "true", "--connection-string", $conn,
        "--query", "[?!isDirectory].{name:name,size:contentLength}", "-o", "json", "--only-show-errors")
    # En PowerShell 5.1 ConvertFrom-Json devuelve el array como un único objeto: se desenrolla.
    $parsed = ($listing -join "`n") | ConvertFrom-Json
    $items = @($parsed | ForEach-Object { $_ } | Where-Object { $null -ne $_ })
    $totalBytes = 0.0
    foreach ($it in $items) { $totalBytes += [double]$it.size }
    if ($items.Count -gt 0) {
        $items | Sort-Object name | Format-Table @{ n = "Fichero"; e = { $_.name } },
            @{ n = "Tamaño (MB)"; e = { "{0:N2}" -f ([double]$_.size / 1MB) }; a = "Right" } -AutoSize | Out-Host
    }
    Write-Host ("  {0} ficheros, {1:N2} MB en total" -f $items.Count, ($totalBytes / 1MB))
    $expected = $Noises.Count * 2  # un fichero por ruido y longitud (100 y 200 rondas)
    if ($items.Count -lt $expected) {
        throw "En $FileSystem/$UploadPrefix hay $($items.Count) ficheros y se esperaban al menos $expected."
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
        Write-Step "g) Borrando $($Created.Count) grupos de contenedores"
        foreach ($name in $Created) {
            try {
                Invoke-Az -Step "borrar $name" -Arguments @(
                    "container", "delete", "-g", $ResourceGroup, "-n", $name, "--yes", "-o", "none", "--only-show-errors") | Out-Null
                Write-Host "  borrado $name"
            } catch {
                Write-Host "  No se pudo borrar ${name}: bórralo a mano con 'az container delete -g $ResourceGroup -n $name --yes'" -ForegroundColor Yellow
            }
        }
    }
    if ($BuildDir -and (Test-Path $BuildDir)) { Remove-Item -Recurse -Force $BuildDir }
    Remove-Variable -Name conn, acrPassword -ErrorAction SilentlyContinue
    $Secrets.Clear()
    Write-Host ("`nDuración total: {0:hh\:mm\:ss}" -f ((Get-Date) - $Start))
}

if ($failed) { exit 1 }
Write-Host "Simulación distribuida completada." -ForegroundColor Green
exit 0
