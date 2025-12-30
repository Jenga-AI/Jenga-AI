#!/bin/bash
"""
JengaHub Docker Build Script

This script provides comprehensive Docker image building with optimization,
multi-platform support, and caching strategies.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
REGISTRY="jengahub"
VERSION="latest"
PUSH=false
PLATFORM="linux/amd64"
CACHE=true
BUILD_ARGS=""
TARGET=""
DOCKERFILE="docker/Dockerfile"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Help function
show_help() {
    cat << EOF
JengaHub Docker Build Script

Usage: $0 [OPTIONS] [IMAGE_TYPE]

IMAGE_TYPES:
    api         Build main API service (default)
    api-cpu     Build CPU-only API service
    worker      Build worker service
    all         Build all images

OPTIONS:
    -r, --registry REGISTRY     Docker registry (default: jengahub)
    -v, --version VERSION       Image version tag (default: latest)
    -p, --push                  Push images to registry after build
    -t, --target TARGET         Build target stage
    -f, --dockerfile FILE       Dockerfile to use
    --platform PLATFORM         Target platform (default: linux/amd64)
    --no-cache                  Disable build cache
    --build-arg KEY=VALUE       Add build argument
    --multi-platform            Build for multiple platforms
    --development               Build development images
    --production                Build optimized production images
    -h, --help                  Show this help message

EXAMPLES:
    # Build main API image
    $0 api

    # Build and push all images
    $0 --push all

    # Build worker image for production
    $0 --production --version v1.2.3 worker

    # Build with custom build args
    $0 --build-arg CUDA_VERSION=11.8 --target production api

    # Build for multiple platforms
    $0 --multi-platform --push api

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -p|--push)
                PUSH=true
                shift
                ;;
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -f|--dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --no-cache)
                CACHE=false
                shift
                ;;
            --build-arg)
                BUILD_ARGS="$BUILD_ARGS --build-arg $2"
                shift 2
                ;;
            --multi-platform)
                PLATFORM="linux/amd64,linux/arm64"
                shift
                ;;
            --development)
                TARGET="app-code"
                VERSION="dev"
                shift
                ;;
            --production)
                TARGET="production"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            api|api-cpu|worker|all)
                IMAGE_TYPE="$1"
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    info "Validating build environment..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if buildx is available for multi-platform builds
    if [[ $PLATFORM == *","* ]]; then
        if ! docker buildx version &> /dev/null; then
            error "Docker buildx is required for multi-platform builds"
            exit 1
        fi
        
        # Ensure buildx builder is set up
        if ! docker buildx inspect multiplatform &> /dev/null; then
            info "Creating multi-platform builder..."
            docker buildx create --name multiplatform --driver docker-container --use
            docker buildx inspect multiplatform --bootstrap
        else
            docker buildx use multiplatform
        fi
    fi
    
    success "Environment validation passed"
}

# Get build context
get_build_context() {
    # Always build from project root
    cd "$(dirname "$0")/.."
    BUILD_CONTEXT="."
    
    info "Build context: $(pwd)"
}

# Build image function
build_image() {
    local image_name=$1
    local dockerfile=$2
    local target_override=$3
    
    local full_image_name="${REGISTRY}/${image_name}:${VERSION}"
    local cache_args=""
    local target_args=""
    
    info "Building image: $full_image_name"
    info "Dockerfile: $dockerfile"
    info "Platform: $PLATFORM"
    
    # Set target if specified
    local build_target=${target_override:-$TARGET}
    if [[ -n "$build_target" ]]; then
        target_args="--target $build_target"
        info "Target stage: $build_target"
    fi
    
    # Configure cache
    if [[ "$CACHE" == "true" ]]; then
        cache_args="--cache-from ${REGISTRY}/${image_name}:cache"
        if [[ "$PUSH" == "true" ]]; then
            cache_args="$cache_args --cache-to ${REGISTRY}/${image_name}:cache,mode=max"
        fi
    else
        cache_args="--no-cache"
    fi
    
    # Build command
    local build_cmd="docker"
    local push_args=""
    
    # Use buildx for multi-platform or if pushing
    if [[ $PLATFORM == *","* ]] || [[ "$PUSH" == "true" ]]; then
        build_cmd="docker buildx"
        if [[ "$PUSH" == "true" ]]; then
            push_args="--push"
        fi
    fi
    
    # Execute build
    log "Executing build command..."
    $build_cmd build \
        --platform "$PLATFORM" \
        --file "$dockerfile" \
        $target_args \
        $cache_args \
        $BUILD_ARGS \
        --tag "$full_image_name" \
        $push_args \
        "$BUILD_CONTEXT"
    
    if [[ $? -eq 0 ]]; then
        success "Successfully built $full_image_name"
        
        # Push if not using buildx with --push
        if [[ "$PUSH" == "true" && $PLATFORM != *","* ]]; then
            log "Pushing $full_image_name..."
            docker push "$full_image_name"
            success "Successfully pushed $full_image_name"
        fi
        
        # Show image info
        if [[ $PLATFORM != *","* ]]; then
            info "Image size: $(docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' | grep "$full_image_name" | awk '{print $2}')"
        fi
    else
        error "Failed to build $full_image_name"
        return 1
    fi
}

# Build specific image types
build_api() {
    log "Building JengaHub API image..."
    build_image "api" "docker/Dockerfile" "production"
}

build_api_cpu() {
    log "Building JengaHub CPU-only API image..."
    build_image "api-cpu" "docker/Dockerfile.cpu" "production"
}

build_worker() {
    log "Building JengaHub Worker image..."
    build_image "worker" "docker/Dockerfile.worker" "production"
}

# Security scan (if tools available)
security_scan() {
    local image_name=$1
    
    if command -v trivy &> /dev/null; then
        info "Running security scan with Trivy..."
        trivy image "${REGISTRY}/${image_name}:${VERSION}" || warn "Security scan found vulnerabilities"
    elif command -v docker &> /dev/null && docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest --version &> /dev/null; then
        info "Running security scan with Trivy (Docker)..."
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:latest image "${REGISTRY}/${image_name}:${VERSION}" || warn "Security scan found vulnerabilities"
    else
        warn "Trivy not available, skipping security scan"
    fi
}

# Clean up old images
cleanup_images() {
    info "Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f || true
    
    # Remove old versions (keep last 5)
    for image_type in api api-cpu worker; do
        local old_images=$(docker images "${REGISTRY}/${image_type}" --format "{{.Tag}}" | grep -v latest | sort -V | head -n -5)
        for tag in $old_images; do
            if [[ -n "$tag" && "$tag" != "$VERSION" ]]; then
                info "Removing old image: ${REGISTRY}/${image_type}:${tag}"
                docker rmi "${REGISTRY}/${image_type}:${tag}" || true
            fi
        done
    done
}

# Generate build report
generate_build_report() {
    local report_file="build-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "build_info": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "version": "$VERSION",
        "registry": "$REGISTRY",
        "platform": "$PLATFORM",
        "target": "$TARGET",
        "dockerfile": "$DOCKERFILE",
        "image_type": "$IMAGE_TYPE"
    },
    "git_info": {
        "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
        "branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
        "tag": "$(git describe --tags --exact-match 2>/dev/null || echo 'none')"
    },
    "system_info": {
        "docker_version": "$(docker --version)",
        "buildx_version": "$(docker buildx version 2>/dev/null || echo 'not available')",
        "host": "$(hostname)",
        "user": "$(whoami)"
    }
}
EOF
    
    info "Build report saved to: $report_file"
}

# Main function
main() {
    log "JengaHub Docker Build Script"
    echo "=" * 50
    
    # Parse command line arguments
    parse_args "$@"
    
    # Set default image type if not specified
    IMAGE_TYPE=${IMAGE_TYPE:-api}
    
    # Validate environment
    validate_environment
    
    # Get build context
    get_build_context
    
    # Build images based on type
    case $IMAGE_TYPE in
        api)
            build_api
            if [[ "$PUSH" == "true" ]]; then
                security_scan "api"
            fi
            ;;
        api-cpu)
            build_api_cpu
            if [[ "$PUSH" == "true" ]]; then
                security_scan "api-cpu"
            fi
            ;;
        worker)
            build_worker
            if [[ "$PUSH" == "true" ]]; then
                security_scan "worker"
            fi
            ;;
        all)
            build_api
            build_api_cpu
            build_worker
            if [[ "$PUSH" == "true" ]]; then
                security_scan "api"
                security_scan "api-cpu"
                security_scan "worker"
            fi
            ;;
        *)
            error "Unknown image type: $IMAGE_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    # Cleanup if requested
    if [[ "$CACHE" == "true" ]]; then
        cleanup_images
    fi
    
    # Generate build report
    generate_build_report
    
    success "Build process completed successfully!"
}

# Run main function with all arguments
main "$@"