$RemoteUser = "apujols"
$RemoteHost = "pirineus3.csuc.cat"
$LocalDir = "C:\Repositoris\treball-final-de-grau"

$files = ssh "$RemoteUser@$RemoteHost" "cat ~/filelist.txt"

foreach ($file in $files) {
    $file = $file.Trim()

    $localPath = Join-Path $LocalDir $file
    $localDirPath = Split-Path $localPath

    if (-not (Test-Path $localDirPath)) {
        New-Item -ItemType Directory -Path $localDirPath -Force | Out-Null
    }

    scp "$RemoteUser@${RemoteHost}:$file" "$localPath"
}
