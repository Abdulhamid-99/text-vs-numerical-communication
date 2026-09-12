#requires -Version 7.2
<# Execute one fresh study. Each numeric receiver has its own network-disabled
container, numeric-only input mounts, and an initially empty output mount.
Existing attempts are retained and never resumed or overwritten. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $StudyName,
    [Parameter(Mandatory)] [string] $InputsPath,
    [ValidateSet('development','heldout')] [string] $Purpose = 'heldout',
    [ValidateRange(1,4096)] [int] $Count = 64,
    [ValidateSet('qwen','llama')] [string[]] $Models = @('qwen','llama'),
    [string] $HubPath = (Join-Path $env:USERPROFILE '.cache\huggingface\hub'),
    [string] $ImageName = 'budgetfair:numeric-relay',
    [ValidateRange(0,15)] [int] $GpuDevice = 0,
    [switch] $SkipBuild
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$RepoRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$BaseImage = 'budgetfair:corrected'
$ExpectedBaseId = 'sha256:2223d46097b6336d58ce4e69b50775584020506cb6fff1b3ee9095db5eaecd5b'
$script:RunImageId = $null
$script:State = $null
$script:LogDirectory = $null
$script:FailureCode = 2
$script:StepNumber = 0
$script:Docker = $null

function Write-JsonFile([string] $Path, $Document) {
    $tmp = "$Path.$([guid]::NewGuid().ToString('N')).tmp"
    [IO.File]::WriteAllText($tmp, ($Document | ConvertTo-Json -Depth 60), [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($tmp, $Path, $true)
}
function Save-State {
    if ($null -ne $script:LogDirectory) { Write-JsonFile (Join-Path $script:LogDirectory 'launcher_status.json') $script:State }
}
function Assert-RegularPath([string] $Path) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if (((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing a linked/reparse input or output path: $cursor"
            }
        }
        $parent = [IO.Directory]::GetParent($cursor)
        if ($null -eq $parent) { break }
        $cursor = $parent.FullName
    }
}
function Assert-RunPath([string] $Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith(($RunRoot + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output path escaped the declared run directory: $full"
    }
    Assert-RegularPath $full
}
function Verify-FrozenInputs {
    $manifestPath = Join-Path $ResolvedInputs 'inputs_manifest.json'
    if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $script:State.inputs_manifest_sha256) { throw 'Input manifest changed during this attempt.' }
    foreach ($entry in $inputManifest.files.GetEnumerator()) {
        $path = Join-Path $ResolvedInputs $entry.Key
        Assert-RegularPath $path
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Frozen input hash differs: $($entry.Key)" }
    }
}
function Invoke-DockerStep([string] $Stage, [string[]] $NativeArguments, [switch] $Capture) {
    $index = $script:StepNumber++
    $stem = '{0:D2}_{1}' -f $index, $Stage
    $script:State.current_stage = $Stage
    $record = [ordered]@{stage=$Stage; status='running'; started_utc=[DateTime]::UtcNow.ToString('o');
        finished_utc=$null; exit_code=$null; arguments=$NativeArguments}
    $script:State.steps += $record
    Write-JsonFile (Join-Path $script:LogDirectory "$stem.command.json") @{executable=$script:Docker; arguments=$NativeArguments}
    Save-State
    Write-Host "[$Stage] Starting."
    $logPath = Join-Path $script:LogDirectory "$stem.log"
    if ($Capture) {
        $lines = @(& $script:Docker @NativeArguments 2>&1)
        $nativeCode = $LASTEXITCODE
        $capturedText = ($lines | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
        [IO.File]::WriteAllText($logPath, $capturedText, [Text.UTF8Encoding]::new($false))
    } else {
        & $script:Docker @NativeArguments 2>&1 | Tee-Object -FilePath $logPath | ForEach-Object { Write-Host $_ }
        $nativeCode = $LASTEXITCODE
    }
    $record.exit_code = $nativeCode
    $record.finished_utc = [DateTime]::UtcNow.ToString('o')
    $record.status = if ($nativeCode -eq 0) { 'completed' } else { 'failed' }
    Save-State
    if ($nativeCode -ne 0) {
        $script:FailureCode = $nativeCode
        if ($Capture) { Write-Host $capturedText }
        throw "Docker stage $Stage failed with exit code $nativeCode; later stages were not run."
    }
    if ($Capture) { return $capturedText }
}
function Container-Arguments([switch] $Gpu) {
    $native = @('run','--rm','--pull=never','--network','none','--read-only',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=4g','--shm-size','1g','--entrypoint','python',
        '-e','PYTHONPATH=/code/src','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','HF_HUB_CACHE=/modelhub','-e','HF_HOME=/tmp/huggingface','-e','XDG_CACHE_HOME=/tmp/cache',
        '-e','TORCH_HOME=/tmp/torch','-e','TRITON_CACHE_DIR=/tmp/triton','-e','NUMBA_CACHE_DIR=/tmp/numba',
        '-e','CUDA_CACHE_PATH=/tmp/cuda-cache','-e','TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor')
    if ($Gpu) { $native += @('--gpus',"device=$GpuDevice",'-e','CUDA_VISIBLE_DEVICES=0',
        '--mount',"type=bind,source=$ResolvedHub,target=/modelhub,readonly") }
    else { $native += @('-e','CUDA_VISIBLE_DEVICES=-1') }
    return $native
}

try {
    if (-not $IsWindows) { throw 'Use PowerShell7.2+ on Windows.' }
    if ($StudyName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$' -or $StudyName -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$') { throw 'StudyName must be a fresh safe single directory name.' }
    $StudiesRoot = Join-Path $RepoRoot 'vector_solution\studies'
    $RunRoot = [IO.Path]::GetFullPath((Join-Path $StudiesRoot $StudyName))
    Assert-RegularPath $RunRoot
    if (Test-Path -LiteralPath $RunRoot) { throw "Refusing existing attempt: $RunRoot" }
    $script:LogDirectory = Join-Path $RunRoot 'launcher'
    [IO.Directory]::CreateDirectory($script:LogDirectory) | Out-Null
    $script:State = [ordered]@{schema_version=1; status='running'; current_stage='preflight'; purpose=$Purpose;
        count=$Count; models=$Models; started_utc=[DateTime]::UtcNow.ToString('o'); finished_utc=$null;
        error=$null; image_id=$null; steps=@(); launcher_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $script:LogDirectory 'launcher_source.ps1')
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'Dockerfile.numeric') -Destination (Join-Path $script:LogDirectory 'Dockerfile.numeric')
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'Dockerfile.numeric.dockerignore') -Destination (Join-Path $script:LogDirectory 'Dockerfile.numeric.dockerignore')
    Save-State
    if ($Purpose -eq 'heldout' -and $Count -ne 64) { throw 'Heldout study requires exactly64 new cases.' }
    if (@($Models | Select-Object -Unique).Count -ne $Models.Count) { throw 'Models must not repeat.' }
    if ($ImageName -notmatch '^[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_.-]+$' -or $ImageName -eq $BaseImage) { throw 'Use a distinct safe local image tag.' }
    $script:Docker = (Get-Command docker -ErrorAction Stop).Source
    $ResolvedInputs = (Resolve-Path -LiteralPath $InputsPath -ErrorAction Stop).Path
    $ResolvedHub = (Resolve-Path -LiteralPath $HubPath -ErrorAction Stop).Path
    Assert-RegularPath $ResolvedInputs
    $requiredInputs = @('forbidden_values.json','qwen.yaml','qwen_pca.npz','qwen_lineage.json','llama.yaml','llama_pca.npz','llama_lineage.json')
    foreach ($name in @('inputs_manifest.json') + $requiredInputs) {
        if (-not (Test-Path -LiteralPath (Join-Path $ResolvedInputs $name) -PathType Leaf)) { throw "Missing frozen input: $name" }
    }
    $inputManifest = Get-Content -LiteralPath (Join-Path $ResolvedInputs 'inputs_manifest.json') -Raw | ConvertFrom-Json -AsHashtable
    if (@(Compare-Object ($requiredInputs | Sort-Object) (@($inputManifest.files.Keys) | Sort-Object)).Count -ne 0) { throw 'Input manifest must cover exactly all required frozen inputs.' }
    foreach ($entry in $inputManifest.files.GetEnumerator()) {
        if ([IO.Path]::GetFileName($entry.Key) -ne $entry.Key) { throw 'Unsafe frozen input filename.' }
        if ((Get-FileHash -LiteralPath (Join-Path $ResolvedInputs $entry.Key) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Frozen input hash differs: $($entry.Key)" }
    }
    $script:State.inputs_manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $ResolvedInputs 'inputs_manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    Copy-Item -LiteralPath (Join-Path $ResolvedInputs 'inputs_manifest.json') -Destination (Join-Path $script:LogDirectory 'inputs_manifest.json')
    Verify-FrozenInputs
    Save-State
    $baseId = (Invoke-DockerStep 'inspect_base' @('image','inspect',$BaseImage,'--format','{{.Id}}') -Capture).Trim()
    if ($baseId -ne $ExpectedBaseId) { throw 'Installed dependency image differs from verified pinned base.' }
    if (-not $SkipBuild) {
        # BuildKit treats a bare local sha256 ID in FROM as an image repository.
        # Use the verified local tag, then prove the resulting image's layer prefix.
        Invoke-DockerStep 'build' @('build','--network=none','--pull=false','--build-arg',"BASE_IMAGE=$BaseImage",'-f',(Join-Path $RepoRoot 'Dockerfile.numeric'),'-t',$ImageName,$RepoRoot)
    }
    $script:RunImageId = (Invoke-DockerStep 'inspect_image' @('image','inspect',$ImageName,'--format','{{.Id}}') -Capture).Trim()
    $script:State.image_id = $script:RunImageId
    $baseLayers = (Invoke-DockerStep 'inspect_base_layers' @('image','inspect',$baseId,'--format','{{json .RootFS.Layers}}') -Capture) | ConvertFrom-Json
    $runLayers = (Invoke-DockerStep 'inspect_image_layers' @('image','inspect',$script:RunImageId,'--format','{{json .RootFS.Layers}}') -Capture) | ConvertFrom-Json
    if ($runLayers.Count -lt $baseLayers.Count) { throw 'Run image has fewer layers than its pinned dependency base.' }
    for ($index=0; $index -lt $baseLayers.Count; $index++) {
        if ($baseLayers[$index] -ne $runLayers[$index]) { throw 'Run image dependency layers differ from the pinned base.' }
    }
    $files = [ordered]@{}
    foreach ($folder in @('src','scripts','tests')) {
        foreach ($item in Get-ChildItem -LiteralPath (Join-Path $RepoRoot $folder) -Filter '*.py' -File -Recurse | Sort-Object FullName) {
            $relative = [IO.Path]::GetRelativePath($RepoRoot,$item.FullName).Replace('\','/')
            $files[$relative] = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $files['pyproject.toml'] = (Get-FileHash -LiteralPath (Join-Path $RepoRoot 'pyproject.toml') -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-JsonFile (Join-Path $script:LogDirectory 'image_expected_files.json') $files
    $verifyCode = 'import hashlib,json,pathlib,budgetfair; root=pathlib.Path("/code"); expected=json.load(open("/expected.json")); actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ("src","scripts","tests") for p in (root/d).rglob("*.py")}; actual["pyproject.toml"]=hashlib.sha256((root/"pyproject.toml").read_bytes()).hexdigest(); assert actual==expected,"Image source/test inventory differs"; assert pathlib.Path(budgetfair.__file__).is_relative_to(root),"Wrong package imported"; print(json.dumps({"status":"verified","file_count":len(actual),"package_path":budgetfair.__file__}))'
    $native = @(Container-Arguments) + @('--mount',"type=bind,source=$(Join-Path $script:LogDirectory 'image_expected_files.json'),target=/expected.json,readonly",$script:RunImageId,'-c',$verifyCode)
    Invoke-DockerStep 'verify_image_source' $native
    $prepare = @('/code/scripts/run_numeric_relay_study.py','prepare','--study','/run/study','--forbidden-values','/inputs/forbidden_values.json',
        '--purpose',$Purpose,'--count',"$Count",'--pca-dim','512','--codec','int4','--qwen-layer','4','--llama-layer','0','--models') + $Models
    foreach ($model in $Models) {
        $pcaCodec = if ($model -eq 'qwen') { 'int4' } else { 'int8' }
        $prepare += @("--$model-config","/inputs/$model.yaml","--$model-pca-asset","/inputs/${model}_pca.npz", "--$model-pca-lineage","/inputs/${model}_lineage.json","--$model-pca-codec",$pcaCodec)
    }
    $native = @(Container-Arguments) + @('--mount',"type=bind,source=$ResolvedInputs,target=/inputs,readonly",'--mount',"type=bind,source=$RunRoot,target=/run",$script:RunImageId) + $prepare
    Verify-FrozenInputs
    Invoke-DockerStep 'prepare' $native
    $StudyPath = Join-Path $RunRoot 'study'
    foreach ($model in $Models) {
        $native = @(Container-Arguments -Gpu) + @('--mount',"type=bind,source=$StudyPath,target=/study",$script:RunImageId,'/code/scripts/run_numeric_relay_study.py','send','--study','/study','--model',$model)
        Invoke-DockerStep "send_$model" $native
    }
    $PublicPath = Join-Path $StudyPath 'public'
    $ReceiptsPath = Join-Path $RunRoot 'receipts'
    foreach ($group in @('numeric','text')) {
        foreach ($model in $Models) {
            $outputParent = Join-Path (Join-Path (Join-Path $RunRoot 'receipt_staging') $model) $group
            Assert-RunPath $outputParent
            [IO.Directory]::CreateDirectory($outputParent) | Out-Null
            if (@(Get-ChildItem -LiteralPath $outputParent -Force).Count -ne 0) { throw 'Every receiver output mount must start empty.' }
            $native = @(Container-Arguments -Gpu)
            foreach ($name in @('plan.json','prepare_receipt.json','model_configs','assets')) {
                $native += @('--mount',"type=bind,source=$(Join-Path $PublicPath $name),target=/public/$name,readonly")
            }
            $modelInput = Join-Path (Join-Path $PublicPath 'receiver_inputs') $model
            foreach ($name in @('jobs.json','ready.json','public_prompt.json',$group)) {
                $native += @('--mount',"type=bind,source=$(Join-Path $modelInput $name),target=/public/receiver_inputs/$model/$name,readonly")
            }
            $native += @('--mount',"type=bind,source=$outputParent,target=/receipts/$model",$script:RunImageId)
            $isolationCode = 'import json,pathlib,runpy,sys; group,model=sys.argv[1:3]; other="text" if group=="numeric" else "numeric"; forbidden=["/study","/work","/run/study","/public/private_cases.json","/public/sender","/public/receiver_inputs/"+model+"/"+other,"/receipts/"+model+"/"+other]; assert all(not pathlib.Path(p).exists() for p in forbidden),"Private or other-group path exposed"; parent=pathlib.Path("/receipts")/model; assert not list(parent.iterdir()),"Receiver output mount was not empty"; evidence={"isolation_preflight":"passed","group":group,"model":model,"absent_paths":forbidden,"output_initially_empty":True}; (parent/"isolation_preflight.json").write_text(json.dumps(dict(evidence,mountinfo=pathlib.Path("/proc/self/mountinfo").read_text()),indent=2)); print(json.dumps(evidence),flush=True); sys.argv=["/code/scripts/run_numeric_relay_study.py","receive","--public","/public","--output","/receipts","--model",model,"--group",group]; runpy.run_path(sys.argv[0],run_name="__main__")'
            $native += @('-c',$isolationCode,$group,$model)
            Invoke-DockerStep "receive_${model}_$group" $native
            $sourceGroup = Join-Path $outputParent $group
            $finalParent = Join-Path $ReceiptsPath $model
            $finalGroup = Join-Path $finalParent $group
            Assert-RunPath $sourceGroup
            Assert-RunPath $finalGroup
            if (-not (Test-Path -LiteralPath $sourceGroup -PathType Container) -or (Test-Path -LiteralPath $finalGroup)) { throw 'Completed receipt group is missing or its final destination exists.' }
            [IO.Directory]::CreateDirectory($finalParent) | Out-Null
            Move-Item -LiteralPath $sourceGroup -Destination $finalGroup
        }
    }
    $native = @(Container-Arguments) + @('--mount',"type=bind,source=$RunRoot,target=/run",$script:RunImageId,'/code/scripts/run_numeric_relay_study.py','analyze','--study','/run/study','--receipts','/run/receipts','--output','/run/analysis')
    Invoke-DockerStep 'analyze' $native
    Verify-FrozenInputs
    $report = Get-Content -LiteralPath (Join-Path $RunRoot 'analysis\analysis.json') -Raw | ConvertFrom-Json
    $script:State.scientific_gate_passed = $report.scientific_gate.passed
    $script:State.pca_gate_passed = $report.pca_success_gate.passed
    $script:State.status = 'completed'
    $script:State.current_stage = 'finished'
    $script:State.finished_utc = [DateTime]::UtcNow.ToString('o')
    Save-State
    Write-Host "Completed study: $RunRoot"
    exit 0
} catch {
    if ($null -ne $script:State) {
        $script:State.status = 'failed'
        $script:State.finished_utc = [DateTime]::UtcNow.ToString('o')
        $script:State.error = $_.Exception.Message
        Save-State
    }
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit $script:FailureCode
}
