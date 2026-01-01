#!/bin/bash
# Create Colab-ready package for Jenga-AI

echo "🚀 Creating Colab package for Jenga-AI..."

# Create temporary directory for clean package
TEMP_DIR="/tmp/jenga-ai-colab"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

echo "📂 Copying essential files..."

# Copy essential directories
cp -r multitask_bert $TEMP_DIR/
cp -r llm_finetuning $TEMP_DIR/
cp -r seq2seq_models $TEMP_DIR/
cp -r jengahub $TEMP_DIR/
cp -r tests $TEMP_DIR/
cp -r examples $TEMP_DIR/
cp -r docs $TEMP_DIR/
cp -r scripts $TEMP_DIR/
cp -r k8s $TEMP_DIR/
cp -r docker $TEMP_DIR/

# Copy essential root files
cp setup.py $TEMP_DIR/
cp requirements.txt $TEMP_DIR/
cp requirements_colab.txt $TEMP_DIR/
cp setup_colab.py $TEMP_DIR/
cp README.MD $TEMP_DIR/
cp COLAB_QUICK_START.md $TEMP_DIR/
cp TESTING_IMPLEMENTATION_REPORT.md $TEMP_DIR/
cp TESTING_QUICK_START.md $TEMP_DIR/
cp LICENSE $TEMP_DIR/
cp *.md $TEMP_DIR/ 2>/dev/null || true
cp *.yaml $TEMP_DIR/ 2>/dev/null || true
cp *.yml $TEMP_DIR/ 2>/dev/null || true

echo "🧹 Cleaning up unnecessary files..."

# Remove large/unnecessary files from temp directory
cd $TEMP_DIR

# Remove large documentation assets
rm -rf docs/site docs/docs/site
find docs -name "*.png" -size +1M -delete 2>/dev/null || true
find docs -name "*.jpg" -size +1M -delete 2>/dev/null || true

# Remove test outputs and artifacts
rm -rf tests/outputs tests/mlruns tests/performance
find tests -name "*.log" -delete 2>/dev/null || true

# Remove Python cache and build artifacts
find . -name "__pycache__" -exec rm -rf {} \; 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name ".pytest_cache" -exec rm -rf {} \; 2>/dev/null || true
rm -rf *.egg-info build dist

# Remove large Jupyter notebooks
find . -name "*.ipynb" -size +1M -delete 2>/dev/null || true

# Remove unnecessary Docker/K8s files for Colab
rm -rf docker/Dockerfile.worker docker/worker-*

echo "📦 Creating ZIP package..."

# Create the ZIP file
cd /tmp
zip -r jenga-ai-core.zip jenga-ai-colab/ -q

# Move to home directory or original location
mv jenga-ai-core.zip ~/jenga-ai-core.zip

# Check final size
FINAL_SIZE=$(du -h ~/jenga-ai-core.zip | cut -f1)
echo "✅ Package created: ~/jenga-ai-core.zip ($FINAL_SIZE)"

echo "📊 Package contents:"
cd $TEMP_DIR
du -h --max-depth=1 | sort -hr

echo ""
echo "🎯 Next steps:"
echo "1. Upload ~/jenga-ai-core.zip to Google Drive"
echo "2. Open Google Colab"
echo "3. Follow instructions in COLAB_QUICK_START.md"

# Cleanup
rm -rf $TEMP_DIR

echo "🎉 Package ready for Colab!"