#!/usr/bin/env bash

# Check that there are no std artifacts in no_std builds
cargo build --no-default-features
! find target/ -name *.rlib -exec nm {} \; | grep "std"

cargo test --no-default-features
cargo test --no-default-features --features use_std
cargo test --no-default-features --features unstable
cargo test --no-default-features --features use_std,unstable

cargo test --no-default-features --release
cargo test --no-default-features --release --features use_std,unstable
