#!/usr/bin/env bash

set -ex

: ${TARGET?"The TARGET environment variable must be set."}

export RUST_TEST_THREADS=1
export RUST_BACKTRACE=1
export RUST_TEST_NOCAPTURE=1

# Check that there are no std artifacts in no_std builds
cargo build --no-default-features
! find target/ -name *.rlib -exec nm {} \; | grep "std"

cargo test --no-default-features
cargo test --no-default-features --features use_std
cargo test --no-default-features --release

if [[ $TRAVIS_RUST_VERSION == "nightly" ]] || [[ $TARGET = *"windows"* ]]; then
    cargo test --no-default-features --features unstable
    cargo test --no-default-features --features use_std,unstable
    cargo test --no-default-features --release --features use_std,unstable
    RUSTFLAGS="-C target-cpu=native" cargo test --no-default-features --release --features use_std,unstable
fi
