Write-Host "Installing dependencies..." -ForegroundColor Cyan

Write-Host -NoNewline "`t- Cloning 'CLIP'..." 
git clone "https://github.com/openai/CLIP" --quiet
Write-Host "done" -ForegroundColor Green

Write-Host -NoNewline "`t- Cloning 'Taming Transformers'..."
git clone "https://github.com/CompVis/taming-transformers" --quiet
Write-Host "done" -ForegroundColor Green

Write-Host "Creating directories..."
mkdir "./content/steps" -Force