fn main() {
    // Workaround for stack overflow in tauri_build::build() caused by
    // recursive schema definitions (Value type referring to itself).
    // Spawn a thread with a larger stack (16MB instead of default 8MB).
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024) // 16MB
        .spawn(|| {
            tauri_build::build();
        })
        .expect("failed to spawn build thread")
        .join()
        .expect("build thread panicked");
}
