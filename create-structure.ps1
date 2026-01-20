param(
    [string]$Base = "D:\Personal\deep-learning-template\deep_learning_project"
)

# Directories to create
$dirs = @(
    $Base,
    "$Base\configs",
    "$Base\configs\callbacks",
    "$Base\configs\data",
    "$Base\configs\model",
    "$Base\configs\trainer",
    "$Base\src",
    "$Base\src\data",
    "$Base\src\models",
    "$Base\src\utils",
    "$Base\scripts"
)

# Files to create (empty)
$files = @(
    "$Base\configs\train.yaml",
    "$Base\src\__init__.py",
    "$Base\pyproject.toml",
    "$Base\train.py",
    "$Base\.env"
)

# Create directories
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -Path $d -ItemType Directory -Force | Out-Null
    }
}

# Create files (empty)
foreach ($f in $files) {
    if (-not (Test-Path $f)) {
        New-Item -Path $f -ItemType File -Force | Out-Null
    }
}

Write-Output "Project skeleton created at: $Base"
```// filepath: d:\Personal\deep-learning-template\scripts\create_structure.ps1
param(
    [string]$Base = "D:\Personal\deep-learning-template\deep_learning_project"
)

# Directories to create
$dirs = @(
    $Base,
    "$Base\configs",
    "$Base\configs\callbacks",
    "$Base\configs\data",
    "$Base\configs\model",
    "$Base\configs\trainer",
    "$Base\src",
    "$Base\src\data",
    "$Base\src\models",
    "$Base\src\utils",
    "$Base\scripts"
)

# Files to create (empty)
$files = @(
    "$Base\configs\train.yaml",
    "$Base\src\__init__.py",
    "$Base\pyproject.toml",
    "$Base\train.py",
    "$Base\.env"
)

# Create directories
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -Path $d -ItemType Directory -Force | Out-Null
    }
}

# Create files (empty)
foreach ($f in $files) {
    if (-not (Test-Path $f)) {
        New-Item -Path $f -ItemType File -Force | Out-Null
    }
}

Write-Output "Project skeleton created at: $Base"